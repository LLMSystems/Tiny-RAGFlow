import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.bm25_index import BM25Index


def main():
    parser = argparse.ArgumentParser(description="Search BM25 index")
    
    parser.add_argument(
        "--bm25_config",
        type=str,
        default="./config/bm25.yaml",
        help="Path to BM25 index configuration file"
    )

    parser.add_argument(
        "--query",
        type=str,
        default="hello world",
        help="Search query"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to return"
    )

    args = parser.parse_args()

    # Load index
    index = BM25Index(args.bm25_config, auto_load=True)

    print("\n==== BM25 Search Result ====")

    scores, docs = index.search(args.query, top_k=args.top_k)

    for s, d in zip(scores, docs):
        print(f"{s:.4f}  →  id={d.get('id')}  text={d.get('text')}")


if __name__ == "__main__":
    main()
