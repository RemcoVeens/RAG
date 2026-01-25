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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
