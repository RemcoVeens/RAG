import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import commands
from src.multimodal_search import verify_image_embedding


# from src.hybrid_search import HybridSearch
def main(raw_args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="mulimodal search CLI")
    subparsers = parser.add_subparsers(dest="command")
    search_parser = subparsers.add_parser("verify_image_embedding", help="Search for documents")
    _ = search_parser.add_argument("image", type=str, help="The path to an image file")

    image_search_parser = subparsers.add_parser("image_search", help="Search for documents")
    _ = image_search_parser.add_argument("image", type=str, help="The path to an image file")

    args = parser.parse_args(raw_args)
    match args.command:
        case "verify_image_embedding":
            file_path = Path(args.image)
            if not file_path.exists():
                print(f"Image file does not exist: {file_path}")
            verify_image_embedding(file_path)
            return
        case "image_search":
            file_path = Path(args.image)
            if not file_path.exists():
                print(f"Image file does not exist: {file_path}")
            res = commands.image_search_command(file_path)
            print(res)
            for i, res in enumerate(res, start=1):
                print(f"\n{i}. {res['title']} (similarity: {res['score']:.3f})")
                print(res["description"])
            return


if __name__ == "__main__":
    main()
