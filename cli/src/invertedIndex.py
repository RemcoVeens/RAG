import os
import pickle

from src.classes import Movie


class InvertedIndex:
    def __init__(self):
        # Maps tokens (strings) to sets of document IDs (integers)
        self.index = {}
        # Maps document IDs to the actual movie objects
        self.docmap:dict[int, Movie] = {}

    def __add_document(self, doc_id:int, text:str):
        """
        Tokenizes the input text and adds the doc_id to the
        set of IDs associated with each token.
        """
        # Basic tokenization: lowercase and split by whitespace
        tokens = text.lower().split()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term:str):
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
        os.makedirs('cache', exist_ok=True)

        # Save the index
        with open('cache/index.pkl', 'wb') as f:
            pickle.dump(self.index, f)

        # Save the docmap
        with open('cache/docmap.pkl', 'wb') as f:
            pickle.dump(self.docmap, f)

        print("Index and Docmap saved successfully to 'cache/'.")
