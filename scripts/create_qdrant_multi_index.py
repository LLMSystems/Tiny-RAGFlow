import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import asyncio
import json

from src.core.client.embedding_rerank_client import MultiVectorModel
from src.core.qdrant_index import QdrantIndex
from src.pipelines.qdrant_ingestion import QdrantMultivectorIngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Create a new Qdrant index")
    parser.add_argument(
        "--qdrant_config",
        type=str,
        default="./config/qdrant.yaml",
        help="Path to Qdrant index configuration file",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/dataset.json",
        help="Path to dataset file (JSON format)",
    )
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default="colbert-ir/colbertv2.0",
        help="Embedding model path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for embedding and ingestion",
    )
    
    args = parser.parse_args()

    print("==== Creating Qdrant Index ====")
    index = QdrantIndex(args.qdrant_config, auto_load=False)
    embedder = MultiVectorModel(model_path=args.embedding_model_path)
    pipeline = QdrantMultivectorIngestionPipeline(embedder, index)

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    asyncio.run(pipeline.ingest_dataset(dataset, batch_size=args.batch_size))
    
    print("==== Qdrant Index Created and Data Ingested ====")
    
if __name__ == "__main__":
    main()
    