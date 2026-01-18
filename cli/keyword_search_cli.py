#!/usr/bin/env python3

import argparse
import json

from src import InvertedIndex, Movie, Movies, search


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    _= search_parser.add_argument("query", type=str, help="Search query")

    _ = subparsers.add_parser("build", help="builds movies inverted index")

    tf = subparsers.add_parser("tf", help="counting tokens")
    _= tf.add_argument("doc_id", type=int, help="Document ID to calculate term frequency")
    _= tf.add_argument("token", type=str, help="Token to calculate term frequency")

    idf = subparsers.add_parser("idf", help="calculating inverse document frequency")
    _= idf.add_argument("term", type=str, help="Token to calculate inverse document frequency")

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
            return
        case "tf":
            ii = InvertedIndex()
            try:
                ii.load()
            except FileNotFoundError:
                print("Inverted index not found. Please build it first.")
                return
            doc_id:int = args.doc_id
            token:str = args.token
            tf = ii.get_tf(doc_id, token)
            print(f"Term frequency of '{token}' in document {doc_id}: {tf}")
            return
        case "idf":
            ii = InvertedIndex()
            try:
                ii.load()
            except FileNotFoundError:
                print("Inverted index not found. Please build it first.")
                return
            term:str = args.term
            idf = ii.get_idf(term)
            print(f"Inverse document frequency of '{term}': {idf:.2f}")
            return
        case _:
            parser.print_help()
            return


if __name__ == "__main__":
    from datetime import datetime
    start_time = datetime.now()
    main()
    print(f"Execution time: {datetime.now() - start_time}")
