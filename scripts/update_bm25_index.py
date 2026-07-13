import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json

from src.core.bm25_index import BM25Index
from src.pipelines.bm25_ingestion import BM25IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Update BM25 index with new data")
    parser.add_argument(
        "--bm25_config",
        type=str,
        default="./config/bm25.yaml",
        help="Path to BM25 index configuration file",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/dataset.json",
        help="Path to new dataset file (JSON format)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for ingestion",
    )

    args = parser.parse_args()

    print("==== Updating BM25 Index ====")

    index = BM25Index(args.bm25_config, auto_load=True)

    pipeline = BM25IngestionPipeline(index, auto_fix_id=True)

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    print(f"Loaded {len(new_data)} new items. Starting ingest...")

    pipeline.ingest_dataset(new_data, batch_size=args.batch_size)

    print("\n==== BM25 Index Updated Successfully ====")


if __name__ == "__main__":
    main()
