import argparse
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.bm25_index import BM25Index
from src.pipelines.bm25_ingestion import BM25IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Create a new BM25 index")
    
    parser.add_argument(
        "--bm25_config",
        type=str,
        default="./config/bm25.yaml",
        help="Path to BM25 index configuration file"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/dataset.json",
        help="Path to dataset file (JSON format)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for ingestion (tokenization)"
    )
    
    args = parser.parse_args()

    print("==== Creating BM25 Index ====")

    index = BM25Index(args.bm25_config, auto_load=False)

    pipeline = BM25IngestionPipeline(index)

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    pipeline.ingest_dataset(dataset, batch_size=args.batch_size)

    print("==== BM25 Index Created and Data Ingested ====")


if __name__ == "__main__":
    main()
