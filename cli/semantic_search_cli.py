#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import commands, get_semantic_parcer


def main(raw_args: list[str] | None = None) -> None:
    parser = get_semantic_parcer()
    args = parser.parse_args(raw_args)

    match args.command:
        case "verify":
            commands.verify()
        case "embed_text":
            commands.command_embed_text(args.text)
        case "verify_embeddings":
            commands.command_verify_embeddings()
        case "embedquery":
            commands.command_embed_query(args.query)
        case "search":
            commands.command_semantic_search(args.query, args.limit)
        case "chunk":
            commands.command_chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            commands.command_semantic_chunk(args.text, args.max_chunk_size, args.overlap)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
