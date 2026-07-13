import logging
import pickle
import os
from typing import Dict, List, Optional, Callable, Any

import faiss
import numpy as np
import yaml
from tqdm import tqdm
from .base_index import BaseIndex


class FaissIndex(BaseIndex):
    def __init__(self, config_path: str, auto_load=False):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        super().__init__(config)

        self.dimension = config["dimension"]
        self.index_config = config["index"]
        self.index_path = config["paths"]["index_path"]
        self.metadata_path = config["paths"]["metadata_path"]
        
        self.normalize = self.index_config.get("normalize", True)
        
        self.index = self._build_index()
        self.metadata: List[Dict] = []
        
        if auto_load:
            self.load()
        else:
            self.logger.info(f"Initialized FAISS index: {self.index_config}, normalize={self.normalize}")
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
    
    def _normalize(self, vec: np.ndarray):
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return vec / norm

    def _build_index(self):
        index_type = self.index_config["type"]
        metric = self.index_config.get("metric", "ip")

        # metric = ip / l2
        metric_type = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2

        if index_type == "FlatIP":
            return faiss.IndexFlatIP(self.dimension)

        if index_type == "FlatL2":
            return faiss.IndexFlatL2(self.dimension)

        def build_quantizer():
            if metric == "ip":
                return faiss.IndexFlatIP(self.dimension)
            else:
                return faiss.IndexFlatL2(self.dimension)

        if index_type == "IVF":
            nlist = self.index_config['ivf'].get("nlist", 1024)
            quantizer = build_quantizer()
            index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                nlist,
                metric_type,
            )
            self.logger.info(f"Created IVF index with nlist={nlist}")
            return index

        if index_type == "IVF_PQ":
            nlist = self.index_config['ivfpq'].get("nlist", 1024)
            pq_m = self.index_config['ivfpq'].get("M", 16)

            quantizer = build_quantizer()
            index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                nlist,
                pq_m,  
                8      
            )
            self.logger.info(f"Created IVF-PQ index with nlist={nlist}, pq_m={pq_m}")
            return index

        if index_type == "HNSW":
            M = self.index_config['hnsw'].get("M", 32)
            index = faiss.IndexHNSWFlat(self.dimension, M, metric_type)
            index.hnsw.efSearch = self.index_config['hnsw'].get("efSearch", 32)
            index.hnsw.efConstruction = self.index_config['hnsw'].get("efConstruction", 40)

            self.logger.info(f"Created HNSW index with M={M}")
            return index

        raise ValueError(f"Unsupported index type: {index_type}")

    def add(self, vector: np.ndarray, metadata: Dict):
        if vector.ndim != 2:
            self.logger.error("Vector must be 2-dimensional")
            raise ValueError("Vector must be shape (1, dim)")

        if self.normalize:
            vector = self._normalize(vector)
        
        if isinstance(self.index, faiss.IndexIVF):
            if not self.index.is_trained:
                self.logger.info("Training IVF index...")
                self.index.train(vector)

        self.index.add(vector)
        self.metadata.append(metadata)
        self.logger.info(f"Added vector with metadata: {metadata}")
        
    def add_batch(self, vectors: np.ndarray, metadatas: List[Dict], batch_size: int = 32):
        if vectors.ndim != 2:
            self.logger.error("Vectors must be 2-dimensional")
            raise ValueError("Vectors must be shape (N, dim)")
        if len(vectors) != len(metadatas):
            self.logger.error("Vectors and metadatas length mismatch")
            raise ValueError("Length of vectors and metadatas must be the same")
        n = len(vectors)
        self.logger.info(f"Starting batch ingest: {n} vectors")
        
        if self.normalize:
            vectors = self._normalize(vectors)
        
        if isinstance(self.index, faiss.IndexIVF):
            train_samples = vectors[: min(50000, n)]
            self.index.train(train_samples)
            self.logger.info("Training complete.")
            
        for i in tqdm(range(0, n, batch_size), desc="Adding vectors"):
            batch_vecs = vectors[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]

            self.index.add(batch_vecs)
            self.metadata.extend(batch_meta)

        self.logger.info("Batch ingest completed.")    

    def search(
            self, 
            vector: np.ndarray, 
            top_k: int,
            max_retries: int = 3, 
            expansion_factor: int = 2,
            dedup_key: Optional[str] = None, 
            dedup_fn: Optional[Callable] = None
        ):
        """
        dedup_key : "metadata.A.B.C"  # 支持多層級的 key
        """
        if vector.ndim != 2:
            raise ValueError("Vector must be shape (1, dim)")

        if self.normalize:
            vector = self._normalize(vector)

        base_k = top_k
        attempt = 0

        while attempt < max_retries:
            top_k_search = base_k * (expansion_factor ** attempt)
            scores, ids = self.index.search(vector, top_k_search)

            row_ids = ids[0]
            row_scores = scores[0]
            seen = set()
            unique_scores = []
            docs = []
            for sid, sc in zip(row_ids, row_scores):
                if sid < 0:
                    continue
                dedup_val = self._get_dedup_value(
                    self.metadata[sid], dedup_key, dedup_fn
                )
                if dedup_val is None:
                    try:
                        dedup_val = self.metadata[sid]['text']
                    except Exception:
                        self.logger.warning("No 'text' field in metadata for deduplication. Using id instead.")
                        dedup_val = sid

                try:
                    key = dedup_val if isinstance(dedup_val, (str, int, float, bool)) else hash(dedup_val)
                except Exception:
                    key = str(dedup_val)
                
                if key in seen:
                    continue
                seen.add(key)

                unique_scores.append(sc)
                docs.append(self.metadata[sid])
                if len(docs) >= top_k:
                    break
            if len(docs) >= top_k or attempt == max_retries - 1:
                return np.array(unique_scores), docs

            attempt += 1

        return np.array(unique_scores), docs

    def search_batch(
            self, vectors: np.ndarray, 
            top_k: int,
            max_retries: int = 3,
            expansion_factor: int = 2,
            dedup_key: Optional[str] = None,
            dedup_fn: Optional[Callable] = None
        ):
        if vectors.ndim != 2:
            raise ValueError("Vectors must be shape (N, dim)")

        if self.normalize:
            vectors = self._normalize(vectors)

        base_k = top_k
        attempt = 0
        while attempt < max_retries:
            top_k_search = base_k * (expansion_factor ** attempt)
            scores, ids = self.index.search(vectors, top_k_search)

            all_docs = []
            all_scores = []
            for row_idx in range(len(vectors)):
                row_ids = ids[row_idx]
                row_scores = scores[row_idx]
                seen = set()
                unique_scores = []
                docs = []
                for sid, sc in zip(row_ids, row_scores):
                    if sid < 0:
                        continue
                    dedup_val = self._get_dedup_value(
                        self.metadata[sid], dedup_key, dedup_fn
                    )
                    if dedup_val is None:
                        try:
                            dedup_val = self.metadata[sid]['text']
                        except Exception:
                            self.logger.warning("No 'text' field in metadata for deduplication. Using id instead.")
                            dedup_val = sid

                    try:
                        key = dedup_val if isinstance(dedup_val, (str, int, float, bool)) else hash(dedup_val)
                    except Exception:
                        key = str(dedup_val)
                    
                    if key in seen:
                        continue
                    seen.add(key)

                    unique_scores.append(sc)
                    docs.append(self.metadata[sid])
                    if len(docs) >= top_k:
                        break
                all_docs.append(docs)
                all_scores.append(np.array(unique_scores))
            
            if all(len(docs) >= top_k for docs in all_docs) or attempt == max_retries - 1:
                return all_scores, all_docs

            attempt += 1
        return all_scores, all_docs 
    
    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        self.logger.info(
            f"Index saved to {self.index_path} and metadata saved to {self.metadata_path}"
        )

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        self.logger.info(
            f"Index loaded from {self.index_path} and metadata loaded from {self.metadata_path}"
        )
