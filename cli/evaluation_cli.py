import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.hybrid_search import HybridSearch


def main(raw_args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    _ = parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args(raw_args)
    limit: int = args.limit

    # run evaluation logic here
    with open("data/golden_dataset.json") as f:
        golden = json.load(f)
    for case in golden["test_cases"]:
        query: str = case["query"]
        relevant_docs: list[str] = case["relevant_docs"]
        hs = HybridSearch()
        total_retrieved = hs.rrf_search(query, 60, limit)
        all_retrieved_titles = [retrieved[1].data.title for retrieved in total_retrieved]
        relevant_retrieved = [title for title in all_retrieved_titles if title in relevant_docs]

        precision = len(relevant_retrieved) / len(total_retrieved)
        recall = len(relevant_retrieved) / len(relevant_docs)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"\n- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(all_retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_retrieved)}")


if __name__ == "__main__":
    main()
