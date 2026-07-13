import os
import pickle
import yaml
import numpy as np
from typing import Dict, List, Any, Optional, Callable

from qdrant_client import QdrantClient, models
from .base_index import BaseIndex


class QdrantIndex(BaseIndex):
    def __init__(self, config_path: str, auto_load=False):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        super().__init__(config)

        self.dimension = config['dimension']
        self.index_config = config['index']

        self.storage_path = config['paths']['storage_path']
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        self.collection_name = config['paths']['collection_name']

        self.mode = self.index_config.get("mode", "dense")
        self.normalize = self.index_config.get("normalize", True)
        self.index_type = self.index_config.get("type", "Flat")
        self.distance = self.index_config.get("distance", "COSINE")

        self.client = QdrantClient(path=self.storage_path)

        if not auto_load:
            self._build_index()
            self.logger.info(f"Initialized Qdrant index: {self.index_config}")
        self.logger.info(f"number of points in collection '{self.collection_name}': {self.client.get_collection(self.collection_name).points_count}")

    def _build_index(self):
        index_type = self.index_type
        hnsw_cfg = self.index_config.get("hnsw", {})
        hnsw_config = None
        
        if index_type == "HNSW":
            hnsw_config = models.HnswConfigDiff(
                m=hnsw_cfg.get("m", 16),
                ef_construct=hnsw_cfg.get("ef_construct", 200),
                full_scan_threshold=hnsw_cfg.get("full_scan_threshold", 10000),
                max_indexing_threads=hnsw_cfg.get("max_indexing_threads", 1),
                on_disk=hnsw_cfg.get("on_disk", False),
            )

        if self.mode == "multivector":
            mv_cfg = self.index_config["multivector"]
            mv_comparator = models.MultiVectorComparator[mv_cfg["comparator"]]

            vector_config = models.VectorParams(
                size=self.dimension,
                distance=models.Distance[self.distance],
                hnsw_config=hnsw_config,
                multivector_config=models.MultiVectorConfig(
                    comparator=mv_comparator,
                )
            )
            
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=vector_config
            )
            self.logger.info(f"Created Qdrant multivector collection: {self.collection_name}")
            return

        # Dense mode
        vectors_config = models.VectorParams(
            size=self.dimension,
            distance=models.Distance[self.distance],
            hnsw_config=hnsw_config,
        )

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config
        )
    
    def _normalize(self, vec: np.ndarray):
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return vec / norm
    
    def add(self, vector: np.ndarray, metadata: Dict):
        if self.mode == "dense":
            if vector.ndim != 2:
                raise ValueError("Dense vector must be shape (1, dim)")
            if self.normalize:
                vector = self._normalize(vector)
            final_vec = vector[0].tolist()
        elif self.mode == "multivector":
            if vector.ndim != 2:
                raise ValueError("Multivector must be shape (T, dim)")
            if self.normalize:
                vector = self._normalize(vector)
            final_vec = vector.tolist()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=self.client.get_collection(self.collection_name).points_count + 1,
                    vector=final_vec,
                    payload=metadata,
                )
            ],
        )
        self.logger.info(f"Added length-1 vector with metadata: {metadata}, shape: {vector.shape}")
        

    def add_batch(self, vectors, metadatas: List[Dict], batch_size: int = 32):
        points = []
        current_count = self.client.get_collection(self.collection_name).points_count

        if self.mode == "dense":  
            if vectors.ndim != 2:
                raise ValueError("Vectors must be shape (N, dim)")
            if len(vectors) != len(metadatas):
                raise ValueError("Number of vectors and metadatas must match")
            if self.normalize:
                vectors = self._normalize(vectors)
            for i in range(len(vectors)):
                points.append(
                    models.PointStruct(
                        id=current_count + i + 1,
                        vector=vectors[i].tolist(),
                        payload=metadatas[i]
                    )
                )
        elif self.mode == "multivector":
            if len(vectors) != len(metadatas):
                raise ValueError("Number of vectors and metadatas must match")
            for i, (v, m) in enumerate(zip(vectors, metadatas)):
                if self.normalize:
                    v = self._normalize(v)
                points.append(
                    models.PointStruct(
                        id=current_count + i + 1,
                        vector=v.tolist(),
                        payload=m
                    )
                )
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=batch_size
        )
        self.logger.info(f"Added batch of {len(vectors)} vectors.")

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        allowed_ids: List[int] = None,
        max_retries: int = 3,
        expansion_factor: int = 2,
        dedup_key: Optional[str] = None,
        dedup_fn: Optional[Callable] = None
    ):
        """Search with optional deduplication. Will expand the candidate limit if
        dedup removes items, up to `max_retries` attempts.
        """
        if self.mode == "dense":
            if vector.ndim != 2:
                raise ValueError("Dense vector must be shape (1, dim)")
            q = vector
            if self.normalize:
                q = self._normalize(vector)
            q = q[0].tolist()
        elif self.mode == "multivector":
            if vector.ndim != 2:
                raise ValueError("Multivector must be shape (T, dim)")
            q = vector
            if self.normalize:
                q = self._normalize(vector)
            q = q.tolist()

        base_k = top_k
        attempt = 0
        while attempt < max_retries:
            top_k_search = base_k * (expansion_factor ** attempt)
            qdrant_filter = self._build_id_filter(allowed_ids) if allowed_ids else None

            result = self.client.query_points(
                collection_name=self.collection_name,
                query=q,
                limit=top_k_search,
                with_payload=True,
                query_filter=qdrant_filter,
            ).points

            seen = set()
            scores = []
            payloads = []
            for point in result:
                payload = point.payload or {}
                try:
                    dedup_val = self._get_dedup_value(payload, dedup_key, dedup_fn)
                except Exception:
                    self.logger.debug("_get_dedup_value failed, falling back to payload['text'] or id")
                    dedup_val = None

                if dedup_val is None:
                    dedup_val = payload.get('text') if isinstance(payload, dict) else None
                if dedup_val is None:
                    dedup_val = getattr(point, 'id', None)

                try:
                    key = dedup_val if isinstance(dedup_val, (str, int, float, bool)) else hash(dedup_val)
                except Exception:
                    key = str(dedup_val)

                if key in seen:
                    continue
                seen.add(key)

                scores.append(point.score)
                payloads.append(payload)
                if len(payloads) >= top_k:
                    break

            if len(payloads) >= top_k or attempt == max_retries - 1:
                return scores, payloads

            attempt += 1

        return scores, payloads

    def search_batch(
        self,
        vectors,
        top_k: int,
        allowed_ids_list: List[List[int]] = None,
        max_retries: int = 3,
        expansion_factor: int = 2,
        dedup_key: Optional[str] = None,
        dedup_fn: Optional[Callable] = None
    ):
        """Batch search. For simplicity and to support per-query expansion for
        deduplication, each vector is queried individually (not using query_batch_points).
        """
        if self.mode == "dense":
            if vectors.ndim != 2:
                raise ValueError("Vector must be shape (N, dim)")
            if self.normalize:
                vectors = self._normalize(vectors)
            vectors = [vec.tolist() for vec in vectors]
        elif self.mode == "multivector":
            vectors = [self._normalize(vec) for vec in vectors]
            vectors = [vec.tolist() for vec in vectors]

        if allowed_ids_list is not None and len(allowed_ids_list) != len(vectors):
            raise ValueError("allowed_ids_list must match number of vectors")

        all_scores = []
        all_payloads = []

        for i, vec in enumerate(vectors):
            allowed_ids = allowed_ids_list[i] if allowed_ids_list else None
            base_k = top_k
            attempt = 0
            scores = []
            payloads = []
            while attempt < max_retries:
                top_k_search = base_k * (expansion_factor ** attempt)
                qdrant_filter = self._build_id_filter(allowed_ids) if allowed_ids else None

                result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vec,
                    limit=top_k_search,
                    with_payload=True,
                    query_filter=qdrant_filter,
                ).points

                seen = set()
                scores = []
                payloads = []
                for point in result:
                    payload = point.payload or {}
                    try:
                        dedup_val = self._get_dedup_value(payload, dedup_key, dedup_fn)
                    except Exception:
                        self.logger.debug("_get_dedup_value failed, falling back to payload['text'] or id")
                        dedup_val = None

                    if dedup_val is None:
                        dedup_val = payload.get('text') if isinstance(payload, dict) else None
                    if dedup_val is None:
                        dedup_val = getattr(point, 'id', None)

                    try:
                        key = dedup_val if isinstance(dedup_val, (str, int, float, bool)) else hash(dedup_val)
                    except Exception:
                        key = str(dedup_val)

                    if key in seen:
                        continue
                    seen.add(key)

                    scores.append(point.score)
                    payloads.append(payload)
                    if len(payloads) >= top_k:
                        break

                if len(payloads) >= top_k or attempt == max_retries - 1:
                    break
                attempt += 1

            all_scores.append(scores)
            all_payloads.append(payloads)

        return all_scores, all_payloads
    
    def _build_id_filter(self, allowed_ids: List[int]):
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="id",
                    match=models.MatchAny(any=allowed_ids)
                )
            ]
        )
    
    def save(self):
        self.logger.info("Qdrant index is automatically saved on disk.")

    def load(self):
        self.logger.info("Qdrant index is automatically loaded from disk.")
