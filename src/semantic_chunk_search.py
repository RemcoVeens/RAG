import json
import re
import string
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .semantic_search import SemanticSearch, cosine_similarity


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
            if len(data) == 1 and data[0].endswith(string.punctuation):
                chunks.extend(movie.description)
            else:
                for d in data:
                    if cleaned := d.rstrip().lstrip():
                        chunks.append(cleaned)
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

    def search_chunks(self, query: str, limit: int = 10) -> list[dict[str, str | float | int]]:
        query = query.rstrip().lstrip()
        if not query:
            return []
        embedding = self.generate_embedding(query)
        chunk_score: list[dict[str, np.float32 | int]] = []
        chunk_embeddings = self.load_or_create_chunk_embeddings()
        for chunk_idx, chunk in enumerate(chunk_embeddings):
            similarity = cosine_similarity(embedding, chunk)
            chunk_score.append(
                {"chunk_idx": chunk_idx, "movie_idx": self.chunk_metadata[chunk_idx]["movie_idx"], "score": similarity}
            )
        movie_score: dict[int, np.float32] = {}
        for cs in chunk_score:
            if cs["movie_idx"] not in movie_score:
                movie_score[cs["movie_idx"]] = cs["score"]
            elif movie_score[cs["movie_idx"]] < cs["score"]:
                movie_score[cs["movie_idx"]] = cs["score"]
        srt = sorted(movie_score.items(), key=lambda item: item[1], reverse=True)[:limit]
        results = [
            {
                "id": id,
                "title": self.document_map[id].title,
                "document": self.document_map[id].description[:100],
                "score": round(score, 3),
                "metadata": self.chunk_metadata[id] or {},
            }
            for id, score in srt
        ]
        return results


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
