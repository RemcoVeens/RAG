from dataclasses import dataclass


@dataclass
class Movie:
    id: int
    title: str
    description: str

    @classmethod
    def from_dict(cls, data: dict):
        return cls(id=data["id"], title=data["title"], description=data["description"])


@dataclass
class Movies:
    movies: list[Movie]

    def from_id(self, id: int):
        for movie in self.movies:
            if movie.id == id:
                return movie
        raise ValueError(f"Movie with id {id} not found")
