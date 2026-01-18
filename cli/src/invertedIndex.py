import math
import os
import pickle
import string
from collections import Counter
from pathlib import Path

from nltk.stem import PorterStemmer
from src.classes import Movie
from tqdm import tqdm


def load_stopwords() -> list[str]:
    with open("data/stopwords.txt", "r") as f:
        return f.read().splitlines()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize(text:str)->list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens:list[str] = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words:list[str] = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words:list[str] = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

class InvertedIndex:
    def __init__(self):
        self.index:dict[str, set[int]] = {}
        self.docmap:dict[int, Movie] = {}
        self.cache_folder = Path("cache")
        self.stemmer = PorterStemmer()
        self.counter:Counter[str] = Counter()
        self.stopwords = load_stopwords()

    def __add_document(self, doc_id:int, text:str):
        """
        Tokenizes the input text and adds the doc_id to the
        set of IDs associated with each token.
        """
        tokens:list[str] =[]
        for token in tokenize(text):
            if token not in self.stopwords and token != "":
                tokens.append(token)
                self.counter[token] += 1

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term:str)->list[int]:
        """
        Returns a sorted list of document IDs containing the term.
        """
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id:int, term:str)->int:
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
        ctr:Counter[str] = Counter()
        for word in combined:
            ctr[word] += 1
        return ctr[token]

    def get_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def build(self, movies:list[Movie]):
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
        with open(self.cache_folder/'index.pkl', 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.cache_folder/'docmap.pkl', 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(self.cache_folder/'term_frequencies.pkl', 'wb') as f:
            pickle.dump(self.counter, f)

        print("Index and Docmap saved successfully to 'cache/'.")

    def load(self):
        index_file = self.cache_folder/'index.pkl'
        docmap_file = self.cache_folder/'docmap.pkl'
        frequencie_file = self.cache_folder/'term_frequencies.pkl'
        if not index_file.exists() or not docmap_file.exists() or not frequencie_file.exists():
            raise FileNotFoundError("Index or Docmap file not found.")

        with open(index_file, 'rb') as f:
            self.index = pickle.load(f)
        with open(docmap_file, 'rb') as f:
            self.docmap = pickle.load(f)
        with open(frequencie_file, 'rb') as f:
            self.counter = pickle.load(f)
