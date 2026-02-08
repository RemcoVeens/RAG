import math
import pickle
import string
from collections import Counter
from pathlib import Path

from nltk.stem import PorterStemmer
from tqdm import tqdm

from .classes import Movie
from .settings import Settings


def load_stopwords() -> list[str]:
    with open("data/stopwords.txt") as f:
        return f.read().splitlines()


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens: list[str] = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words: list[str] = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words: list[str] = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Movie] = {}
        self.cache_folder = Path("cache")
        self.stemmer = PorterStemmer()
        self.counter: Counter[str] = Counter()
        self.doc_lengths: dict[int, int] = {}
        self.stopwords = load_stopwords()

    def __add_document(self, doc_id: int, text: str):
        """
        Tokenizes the input text and adds the doc_id to the
        set of IDs associated with each token.
        """
        tokens: list[str] = []
        for token in tokenize(text):
            if token not in self.stopwords and token != "":
                tokens.append(token)
                self.counter[token] += 1
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        total_tokens = 0.0
        if len(self.doc_lengths) == 0:
            return 0.0
        for _, value in self.doc_lengths.items():
            total_tokens += value
        return total_tokens / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        """
        Returns a sorted list of document IDs containing the term.
        """
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        """
        Returns the term frequency of the term in the document.
        """
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("Term must contain exactly one token")
        token = tokens[0]
        movie = self.docmap.get(doc_id)
        if movie is None:
            return 0
        combined = tokenize(f"{movie.title} {movie.description}")
        ctr: Counter[str] = Counter()
        for word in combined:
            ctr[word] += 1
        return ctr[token]

    def get_idf(self, term: str) -> float:
        """
        get the inverse document frequency of the term.
        """
        doc_count = len(self.docmap)
        term_doc_count = self.get_df(term)
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_df(self, term: str) -> float:
        """
        get the document frequency of the term.
        """
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return len(self.index[token])

    def get_tfidf(self, doc_id: int, token: str) -> float:
        """
        get the term frequency-inverse document frequency of the term in the document.
        """
        tf = self.get_tf(doc_id, token)
        idf = self.get_idf(token)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = self.get_df(term)
        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = Settings.BM25_K1, b: float = Settings.BM25_B):
        tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component

    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        bm25 = self.bm25
        scores: list[tuple[int, float]] = []
        for document_id in self.docmap:
            score = 0.0
            for token in tokens:
                score += bm25(document_id, token)
            scores.append((document_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    def build(self, movies: list[Movie]):
        """
        Populates the index and docmap using a list of movie dictionaries.
        """
        for movie in tqdm(movies, desc="Building index"):
            doc_id = movie.id
            self.docmap[doc_id] = movie

            combined_text = f"{movie.title} {movie.description}"
            self.__add_document(doc_id, combined_text)

    def save(self):
        """
        Saves the index and docmap to the 'cache' directory using pickle.
        """
        self.cache_folder.mkdir(exist_ok=True)
        with open(self.cache_folder / "doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)
        with open(self.cache_folder / "index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open(self.cache_folder / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.cache_folder / "term_frequencies.pkl", "wb") as f:
            pickle.dump(self.counter, f)

        print("Index and Docmap saved successfully to 'cache/'.")

    def load(self):
        index_file = self.cache_folder / "index.pkl"
        docmap_file = self.cache_folder / "docmap.pkl"
        doc_length_file = self.cache_folder / "doc_lengths.pkl"
        frequencie_file = self.cache_folder / "term_frequencies.pkl"
        if not index_file.exists() or not docmap_file.exists() or not frequencie_file.exists():
            raise FileNotFoundError("Index or Docmap file not found.")

        with open(index_file, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_file, "rb") as f:
            self.docmap = pickle.load(f)
        with open(doc_length_file, "rb") as f:
            self.doc_lengths = pickle.load(f)
        with open(frequencie_file, "rb") as f:
            self.counter = pickle.load(f)
