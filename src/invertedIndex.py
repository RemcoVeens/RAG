import math
import pickle
import string
from collections import Counter, defaultdict
from pathlib import Path

from nltk.stem import PorterStemmer
from tqdm import tqdm

from .classes import Movie
from .settings import Settings

# --- Module Level Caching ---
# Initialize expensive resources once at module level to avoid overhead in loops
_STEMMER = PorterStemmer()
_STOPWORDS: set[str] = set()
_STOPWORDS_LOADED = False


def load_stopwords() -> set[str]:
    """Loads stopwords once and caches them."""
    global _STOPWORDS, _STOPWORDS_LOADED
    if not _STOPWORDS_LOADED:
        try:
            with open("data/stopwords.txt") as f:
                _STOPWORDS = set(f.read().splitlines())
        except FileNotFoundError:
            _STOPWORDS = set()
        _STOPWORDS_LOADED = True
    return _STOPWORDS


def preprocess_text(text: str) -> str:
    # translate is fast, keep as is
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def tokenize(text: str) -> list[str]:
    """
    Optimized tokenization using cached resources.
    """
    stopwords = load_stopwords()
    # Preprocess
    text = preprocess_text(text)

    # Tokenize, filter, and stem in a single efficient pass
    tokens = []
    for token in text.split():
        if token and token not in stopwords:
            tokens.append(_STEMMER.stem(token))
    return tokens


class InvertedIndex:
    def __init__(self):
        # Index Structure: term -> {doc_id: frequency}
        # This allows O(1) lookup of TF without re-tokenizing documents.
        self.index: dict[str, dict[int, int]] = defaultdict(dict)
        self.docmap: dict[int, Movie] = {}
        self.cache_folder = Path("cache")

        # Stats
        self.counter: Counter[str] = Counter()
        self.doc_lengths: dict[int, int] = {}
        self.avg_doc_length: float = 0.0  # Pre-computed average

    def __get_avg_doc_length(self) -> float:
        total_tokens = 0.0
        if len(self.doc_lengths) == 0:
            return 0.0
        for _, value in self.doc_lengths.items():
            total_tokens += value
        return total_tokens / len(self.doc_lengths)

    def __add_document(self, doc_id: int, text: str):
        """
        Tokenizes text and populates the index with term frequencies.
        """
        tokens = tokenize(text)

        # Count term frequencies for this document locally first
        term_freqs = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)

        # Update global index and stats
        for token, count in term_freqs.items():
            self.counter[token] += count
            self.index[token][doc_id] = count

    def get_documents(self, term: str) -> list[int]:
        """Returns a sorted list of document IDs containing the term."""
        term = term.lower()
        # Stemming is required to match index keys if the input isn't stemmed
        # Assuming input 'term' needs preprocessing:
        stemmed_tokens = tokenize(term)
        if not stemmed_tokens:
            return []

        # If user passed a single word, use the first token
        target = stemmed_tokens[0]
        return sorted(list(self.index.get(target, {}).keys()))

    def get_tf(self, doc_id: int, term: str) -> int:
        """
        Returns the term frequency from the pre-computed index.
        O(1) operation.
        """
        # We assume 'term' is already stemmed/processed or we process it here.
        # For efficiency in tight loops (like BM25), callers usually pass the stemmed token.
        # If we must support raw strings, we check:
        if term not in self.index:
            # Try tokenizing if it's not a direct key match (fallback)
            tokens = tokenize(term)
            if len(tokens) == 1:
                term = tokens[0]
            else:
                return 0

        return self.index.get(term, {}).get(doc_id, 0)

    def get_idf(self, term: str) -> float:
        doc_count = len(self.docmap)
        term_doc_count = self.get_df(term)
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tfidf(self, doc_id: int, token: str) -> float:
        """
        get the term frequency-inverse document frequency of the term in the document.
        """
        tf = self.get_tf(doc_id, token)
        idf = self.get_idf(token)
        return tf * idf

    def get_df(self, term: str) -> int:
        """Returns document frequency (number of docs containing term)."""
        # Assume term is a valid key (stemmed), if not, return 0
        if term in self.index:
            return len(self.index[term])

        # Fallback for raw strings
        tokens = tokenize(term)
        if tokens and tokens[0] in self.index:
            return len(self.index[tokens[0]])
        return 0

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = Settings.BM25_K1, b: float = Settings.BM25_B):
        tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = self.get_df(term)
        # Avoid division by zero issues or log of negative
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def bm25_search(self, query: str, limit: int = 10) -> list[tuple[int, float]]:
        """
        Optimized BM25 Search.
        Iterates only over relevant documents using the inverted index.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)

        k1 = Settings.BM25_K1
        b = Settings.BM25_B
        avgdl = self.avg_doc_length
        N = len(self.docmap)

        for token in query_tokens:
            if token not in self.index:
                continue
            posting_list = self.index[token]
            if isinstance(posting_list, set):
                doc_items = [(doc_id, 1) for doc_id in posting_list]
                df = len(posting_list)
            else:
                doc_items = posting_list.items()
                df = len(posting_list)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf in doc_items:
                doc_len = self.doc_lengths.get(doc_id, 0)

                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))

                scores[doc_id] += idf * (numerator / denominator)

        scores_list = list(scores.items())
        scores_list.sort(key=lambda x: x[1], reverse=True)
        return scores_list[:limit]

    def build(self, movies: list[Movie]):
        self.docmap = {}
        self.index = defaultdict(dict)
        self.doc_lengths = {}
        self.counter = Counter()

        for movie in tqdm(movies, desc="Building index"):
            self.docmap[movie.id] = movie
            combined_text = f"{movie.title} {movie.description}"
            self.__add_document(movie.id, combined_text)

        # Pre-compute average doc length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        else:
            self.avg_doc_length = 0.0

    def save(self):
        self.cache_folder.mkdir(exist_ok=True)

        # Helper to dump data
        def _dump(filename, data):
            with open(self.cache_folder / filename, "wb") as f:
                pickle.dump(data, f)

        _dump("doc_lengths.pkl", self.doc_lengths)
        _dump("index.pkl", self.index)
        _dump("docmap.pkl", self.docmap)
        _dump("term_frequencies.pkl", self.counter)
        _dump("avg_doc_length.pkl", self.avg_doc_length)

        print("Index and Docmap saved successfully to 'cache/'.")

    def load(self):
        try:
            with open(self.cache_folder / "index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open(self.cache_folder / "docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.cache_folder / "doc_lengths.pkl", "rb") as f:
                self.doc_lengths = pickle.load(f)
            with open(self.cache_folder / "term_frequencies.pkl", "rb") as f:
                self.counter = pickle.load(f)

            # Load avg_doc_length if exists, else recompute
            avg_path = self.cache_folder / "avg_doc_length.pkl"
            if avg_path.exists():
                with open(avg_path, "rb") as f:
                    self.avg_doc_length = pickle.load(f)
            else:
                # Fallback for backward compatibility
                if self.doc_lengths:
                    self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

        except FileNotFoundError:
            raise FileNotFoundError("Index cache files not found. Please build the index first.")
