import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import asyncio
import json
from src.core.qdrant_index import QdrantIndex
from src.core.client.embedding_rerank_client import EmbeddingModel
from src.pipelines.qdrant_ingestion import QdrantIngestionPipeline

def main():
    parser = argparse.ArgumentParser(description="Update Qdrant index with new data")
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
        "--embedding_model",
        type=str,
        default="m3e-base",
        help="Embedding model name",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./config/models.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for embedding and ingestion",
    )
    
    args = parser.parse_args()
    print("==== Updating Qdrant Index ====")

    index = QdrantIndex(args.qdrant_config, auto_load=True)
    embedder = EmbeddingModel(embedding_model=args.embedding_model, config_path=args.model_config_path)
    pipeline = QdrantIngestionPipeline(embedder, index)

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        new_data = json.load(f)
        
    print(f"Loaded {len(new_data)} new items. Starting ingest...")
        
    asyncio.run(pipeline.ingest_dataset(new_data, batch_size=args.batch_size))

    print("\n==== Qdrant Index Updated Successfully ====")


if __name__ == "__main__":
    main()