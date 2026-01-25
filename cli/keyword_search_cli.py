#!/usr/bin/env python3

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import Movie, Movies, commands, get_keyword_parser


def main(raw_args: list[str] | None = None) -> None:
    parser = get_keyword_parser()
    args = parser.parse_args(raw_args)
    movies = [Movie.from_dict(movie) for movie in json.load(open("data/movies.json"))["movies"]]
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
        case "bm25idf":
            _ = commands.command_bm25idf(args.term)
        case "bm25tf":
            _ = commands.command_bm25tf(args.doc_id, args.term, args.k1, args.b)
        case "bm25search":
            _ = commands.command_bm25search(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    from datetime import datetime

    start_time = datetime.now()
    main()
    print(f"Execution time: {datetime.now() - start_time}")
