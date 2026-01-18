#!/usr/bin/env python3

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import InvertedIndex, Movie, Movies, commands, search


def get_parser():
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

    tfidf = subparsers.add_parser("tfidf", help="calculating term frequency inverse document frequency")
    _= tfidf.add_argument("doc_id", type=int, help="Document ID to calculate term frequency inverse document frequency")
    _= tfidf.add_argument("token", type=str, help="Token to calculate term frequency inverse document frequency")
    return parser



def main(args:argparse.ArgumentParser|None=None) -> None:
    parser = get_parser()
    args = parser.parse_args(args)
    movies = [Movie.from_dict(movie) for movie in json.load(open("data/movies.json", "r"))["movies"]]
    MVS = Movies(movies)
    match args.command:
        case "search":
            commands.command_search(args.query, MVS)
        case "build":
            commands.command_build(movies)
        case "tf":
            commands.command_tf(args.doc_id, args.token)
        case "idf":
            commands.command_idf(args.term)
        case "tfidf":
            commands.command_tfidf(args.doc_id, args.token)
        case _:
            parser.print_help()


if __name__ == "__main__":
    from datetime import datetime
    start_time = datetime.now()
    main()
    print(f"Execution time: {datetime.now() - start_time}")
