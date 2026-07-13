import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import asyncio

import numpy as np

from src.core.client.embedding_rerank_client import MultiVectorModel
from src.core.qdrant_index import QdrantIndex


def main():
    parser = argparse.ArgumentParser(description="Search Qdrant index")
    parser.add_argument("--qdrant-config", default="./config/qdrant.yaml")
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default="colbert-ir/colbertv2.0",
        help="Embedding model name",
    )
    parser.add_argument("--query", type=str, default="hello world")
    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    index = QdrantIndex(args.qdrant_config, auto_load=True)
    embedder = MultiVectorModel(model_path=args.embedding_model_path)
    query_vec = asyncio.run(embedder.embed_query(args.query))
    scores, docs = index.search(query_vec, top_k=args.top_k)
    print("\n==== Search Result ====")
    for s, d in zip(scores, docs):
        print(f"{s:.4f}  →  {d}")
        
if __name__ == "__main__":
    main()        