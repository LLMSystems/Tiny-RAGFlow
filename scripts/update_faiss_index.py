import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import asyncio
import json
from src.core.faiss_index import FaissIndex
from src.core.client.embedding_rerank_client import EmbeddingModel
from src.pipelines.faiss_ingestion import FaissIngestionPipeline

def main():
    parser = argparse.ArgumentParser(description="Update FAISS index with new data")
    parser.add_argument(
        "--faiss_config",
        type=str,
        default="./config/faiss.yaml",
        help="Path to FAISS index configuration file",
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
    print("==== Updating FAISS Index ====")
    
    index = FaissIndex(args.faiss_config, auto_load=True)
    embedder = EmbeddingModel(embedding_model=args.embedding_model, config_path=args.model_config_path)
    pipeline = FaissIngestionPipeline(embedder, index, auto_fix_id=True)
    
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        new_data = json.load(f)
        
    print(f"Loaded {len(new_data)} new items. Starting ingest...")
        
    asyncio.run(pipeline.ingest_dataset(new_data, batch_size=args.batch_size))
    
    print("\n==== FAISS Index Updated Successfully ====")
    
    
if __name__ == "__main__":
    main()