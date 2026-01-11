#!/usr/bin/env python3

import argparse
import json

from src import InvertedIndex, Movies, Movie, search


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    _ = subparsers.add_parser("build", help="builds movies inverted index")
    args = parser.parse_args()

    movies = [Movie.from_dict(movie) for movie in json.load(open("data/movies.json", "r"))["movies"]]
    MVS = Movies(movies)
    match args.command:
        case "search":
            ii = InvertedIndex()
            try:
                ii.load()
            except FileNotFoundError:
                print("Inverted index not found. Please build it first.")
                return
            query:str = args.query
            search(ii, query, MVS)
            return
        case "build":
            ii = InvertedIndex()
            ii.build(movies)
            ii.save()
            # docs = ii.get_documents("merida")
            # print(f"First document for token 'merida' = {docs[0]}")
            return
        case _:
            parser.print_help()
            return


if __name__ == "__main__":
    from datetime import datetime
    start_time = datetime.now()
    main()
    print(f"Execution time: {datetime.now() - start_time}")
