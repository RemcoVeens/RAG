import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from . import Movie


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: np.ndarray | None = None
        self.documents: list[Movie] = []
        self.document_map: dict[int, Movie] = {}

    def verify(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")

    def generate_embedding(self, text: str):
        if not text:
            raise ValueError("Text cannot be empty")
        return self.model.encode(text)

    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        entries: list[str] = []
        for doc in documents:
            self.document_map[doc.id] = doc
            entries.append(f"{doc.title}: {doc.description}")
        self.embeddings = self.model.encode(entries, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self):
        movies = [Movie.from_dict(movie) for movie in json.load(open("data/movies.json"))["movies"]]
        self.documents = movies
        for doc in self.documents:
            self.document_map[doc.id] = doc
        if Path("cache/movie_embeddings.npy").exists():
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(self.documents)

    def search(self, query, limit) -> list[dict[str, np.float32 | str]]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        current_query = self.generate_embedding(query)
        distances = cosine_similarity(self.embeddings, current_query)
        simularity_score: list[tuple[np.float32, Movie]] = []
        for i in range(len(distances)):
            simularity_score.append((distances[i], self.documents[i]))
        simularity_score.sort(key=lambda x: x[0], reverse=True)

        return [
            {"score": score, "title": doc.title, "description": doc.description} for score, doc in simularity_score[:limit]
        ]


def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    ss = SemanticSearch()
    _ = ss.load_or_create_embeddings()
    print(f"Number of docs:   {len(ss.documents)}")
    print(f"Embeddings shape: {ss.embeddings.shape[0]} vectors in {ss.embeddings.shape[1]} dimensions")


def embed_query(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2) -> npt.NDArray[np.float32]:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search(query: str, limit: int):
    ss = SemanticSearch()
    _ = ss.load_or_create_embeddings()
    res = ss.search(query, limit)
    for c, result in enumerate(res, start=1):
        print(f"{c}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description']}")
        print()
