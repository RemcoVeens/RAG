from dataclasses import dataclass


@dataclass
class Movie:
    id: int
    title: str
    description: str

    @classmethod
    def from_dict(cls, data: dict):
        return cls(id=data["id"], title=data["title"], description=data["description"])
