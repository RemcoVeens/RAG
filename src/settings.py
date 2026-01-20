from dataclasses import dataclass


@dataclass
class Settings:
    BM25_K1 = 1.5
    BM25_B = 0.75
