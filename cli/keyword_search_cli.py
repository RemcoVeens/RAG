#!/usr/bin/env python3

import argparse
import json
import string
from dataclasses import dataclass

from nltk.stem import PorterStemmer


@dataclass
class Movie:
    id: int
    title: str
    description: str

    @classmethod
    def from_dict(cls, data: dict):
        return cls(id=data["id"], title=data["title"], description=data["description"])
def prep_input(input:str)->str:
    temp = input.lower().strip(string.punctuation).replace(",","")
    return temp

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            query:str = args.query
            print(f"Searching for: {query}")
        case _:
            parser.print_help()
            return
    movies = [Movie.from_dict(movie) for movie in json.load(open("data/movies.json", "r"))["movies"]]
    matches:list[Movie] = []
    stemmer = PorterStemmer()
    with open("data/stopwords.txt", "r") as f:
        stopwords:list[str] = f.read().splitlines()
    tokens = query.split()
    tokens = [token for token in tokens if token not in stopwords]
    for token in tokens:
        for movie in movies:
            if stemmer.stem(token) in prep_input(movie.title):
                if movie not in matches:
                    matches.append(movie)
    for movie in matches:
        print(f"{movie.id}. {movie.title}")

if __name__ == "__main__":
    from datetime import datetime
    start_time = datetime.now()
    main()
    print(f"Execution time: {datetime.now() - start_time}")
