from dataclasses import dataclass


@dataclass
class Movie:
    id: int
    title: str
    description: str

    @classmethod
    def from_dict(cls, data: dict):
        return cls(id=data["id"], title=data["title"], description=data["description"])

    def __str__(self):
        return f"{self.title} ({self.id})"

    def __repr__(self):
        return f"Movie(id={self.id}, title='{self.title}')"


@dataclass
class Movies:
    movies: list[Movie]

    def from_id(self, id: int):
        for movie in self.movies:
            if movie.id == id:
                return movie
        raise ValueError(f"Movie with id {id} not found")


@dataclass
class ChunkResult:
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, str]


@dataclass
class CombinedResults:
    bm25_score: int
    semantic_score: float
    data: ChunkResult
    hybrid_score: float
