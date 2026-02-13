import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    _ = parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit: int = args.limit

    # run evaluation logic here
    with open("data/golden_dataset.json") as f:
        golden = json.load(f)
    for case in golden["test_cases"]:
        query: str = case.get("query")
        total_retrieved = HybridSearch().rrf_search(query, 60, limit)
        titles = [retrieved[1].data.title for retrieved in total_retrieved]
        relevant_retrieved = [title for title in titles if title in case["relevant_docs"]]

        precision = len(relevant_retrieved) / len(total_retrieved)
        print(f"\n- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {','.join(titles)}")
        print(f"  - Relevant: {','.join(relevant_retrieved)}")


if __name__ == "__main__":
    main()
