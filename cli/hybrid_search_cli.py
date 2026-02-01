import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import commands
from src.parcer import get_hybrid_parcer


def main(raw_args: list[str] | None = None) -> None:
    parser = get_hybrid_parcer()
    args = parser.parse_args(raw_args)

    match args.command:
        case "normalize":
            _ = commands.command_hybrid_normalize(args.values)
        case "weighted-search":
            commands.command_weighted_search(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
