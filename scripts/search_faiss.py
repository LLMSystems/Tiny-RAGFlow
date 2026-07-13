import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import asyncio

import numpy as np

from src.core.client.embedding_rerank_client import EmbeddingModel
from src.core.faiss_index import FaissIndex


def main():
    parser = argparse.ArgumentParser(description="Search FAISS index")
    parser.add_argument("--faiss-config", default="./config/faiss.yaml")
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
    parser.add_argument("--query", type=str, default="hello world")
    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    index = FaissIndex(args.faiss_config, auto_load=True)
    embedder = EmbeddingModel(embedding_model=args.embedding_model, config_path=args.model_config_path)
    query_vec = asyncio.run(embedder.embed_documents([args.query]))
    query_vec = np.array(query_vec)
    scores, docs = index.search(query_vec, top_k=args.top_k)
    print("\n==== Search Result ====")
    for s, d in zip(scores, docs):
        print(f"{s:.4f}  →  {d}")
        
if __name__ == "__main__":
    main()        
if __name__ == "__main__":
    main()