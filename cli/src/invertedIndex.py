import os
import pickle
import string
from pathlib import Path
from nltk.stem import PorterStemmer

from src.classes import Movie


class InvertedIndex:
    def __init__(self):
        # Maps tokens (strings) to sets of document IDs (integers)
        self.index = {}
        # Maps document IDs to the actual movie objects
        self.docmap:dict[int, Movie] = {}
        self.cache_folder = Path("cache")
        self.stemmer = PorterStemmer()
        with open("data/stopwords.txt", "r") as f:
            self.stopwords:list[str] = f.read().splitlines()

    def __add_document(self, doc_id:int, text:str):
        """
        Tokenizes the input text and adds the doc_id to the
        set of IDs associated with each token.
        """
        # Basic tokenization: lowercase and split by whitespace
        tokens = text.lower().split()
        tokens = [token.strip(string.punctuation) for token in tokens]
        tokens = [self.stemmer.stem(token)
            for token in tokens
            if token not in self.stopwords and token != ""]
        if doc_id in (167,8):
            print(tokens)

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

    def build(self, movies:list[Movie]):
        """
        Populates the index and docmap using a list of movie dictionaries.
        """
        for movie in movies:
            doc_id = movie.id
            # Store the full object in the docmap
            self.docmap[doc_id] = movie

            # Concatenate title and description for indexing
            combined_text = f"{movie.title} {movie.description}"
            self.__add_document(doc_id, combined_text)

    def save(self):
        """
        Saves the index and docmap to the 'cache' directory using pickle.
        """
        # Create the directory if it doesn't exist
        self.cache_folder.mkdir(exist_ok=True)

        # Save the index
        with open(self.cache_folder/'index.pkl', 'wb') as f:
            pickle.dump(self.index, f)

        # Save the docmap
        with open(self.cache_folder/'docmap.pkl', 'wb') as f:
            pickle.dump(self.docmap, f)

        print("Index and Docmap saved successfully to 'cache/'.")

    def load(self):
        index_file = self.cache_folder/'index.pkl'
        docmap_file = self.cache_folder/'docmap.pkl'
        if not index_file.exists() or not docmap_file.exists():
            raise FileNotFoundError("Index or Docmap file not found.")

        with open(index_file, 'rb') as f:
            self.index = pickle.load(f)
        with open(docmap_file, 'rb') as f:
            self.docmap = pickle.load(f)
