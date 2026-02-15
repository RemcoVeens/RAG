import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import commands
from src.parcer import get_augmented_generation_parcer


def main(raw_args: list[str] | None = None):
    parser = get_augmented_generation_parcer()
    args = parser.parse_args(raw_args)

    match args.command:
        case "rag":
            commands.command_rag(args.query)
            # do RAG stuff here
        case "summarize":
            commands.command_summarize(args.query, args.limit)
        case "citations":
            commands.command_citations(args.query, args.limit)
        case "question":
            commands.command_questions(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
