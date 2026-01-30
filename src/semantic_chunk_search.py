import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .semantic_search import SemanticSearch


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self):
        self.load_movies()
        for doc in self.documents:
            self.document_map[doc.id] = doc
        chunks: list[str] = []
        metas: list[dict[str, int]] = []
        for movie in self.documents:
            if not movie.description:
                continue
            data = semantic_chunk(movie.description, 4, 1)
            chunks.extend(data)
            metas.extend(
                [{"movie_idx": movie.id, "chunk_idx": idx, "total_chunks": len(data)} for idx, _ in enumerate(data)]
            )

        self.chunk_embeddings = [self.generate_embedding(c) for c in tqdm(chunks, desc="Generating embeddings", leave=False)]
        self.chunk_metadata = metas
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self) -> np.ndarray:
        self.load_movies()
        for doc in self.documents:
            self.document_map[doc.id] = doc
        cache_path = Path("cache")
        if (cache_path / "chunk_embeddings.npy").exists() and (cache_path / "chunk_metadata.json").exists():
            self.chunk_embeddings = np.load(cache_path / "chunk_embeddings.npy")
            with open(cache_path / "chunk_metadata.json") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
                self.total_chunks = data["total_chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings()


def semantic_chunk(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        if len(sentences[i:]) <= overlap:
            break
        chunk_slice = sentences[i : i + max_chunk_size]
        chunks.append(" ".join(chunk_slice))
        i += max(1, max_chunk_size - overlap)
        if i >= len(sentences):
            break
    return chunks
