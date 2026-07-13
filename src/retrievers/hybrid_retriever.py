import asyncio
from typing import Any, Dict, List, Optional

from .base_retriever import BaseRetriever
from .retriever_registry import RETRIEVER_REGISTRY


class HybridRetriever(BaseRetriever):
    retriever_type = "hybrid"
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        fusion_method: str = "rrf",
        weights: Optional[List[float]] = None,
        rrf_k: int = 60,
        top_k: int = 5,
    ):
        super().__init__(top_k=top_k)
        self.retrievers = retrievers
        if len(retrievers) < 2:
            raise ValueError("HybridRetriever requires at least two retrievers.")
        
        self.fusion_method = fusion_method
        self.weights = weights
        self.rrf_k = rrf_k
        
        if self.fusion_method == "weighted":
            if weights is None or len(weights) != len(retrievers):
                raise ValueError("Weights must be provided and match the number of retrievers for weighted fusion.")
        self.logger.info("HybridRetriever initialized.")
        
    @classmethod
    def from_config(cls, config: Dict):
        """
        hybrid_config = {
        "retrievers": [
                {
                    "type": "faiss",
                    "config": {
                        "index_config": "./config/faiss.yaml",
                        "embedding_model": "m3e-base",
                        "model_config_path": "./config/models.yaml",
                        "top_k": 3
                    }
                },
                {
                    "type": "bm25",
                    "config": {
                        "index_config": "./config/bm25.yaml",
                        "top_k": 3
                    }
                }
            ],
            "fusion_method": "rrf",
            "rrf_k": 60,
            "top_k": 5
        }
        """
        retrievers_cfg = config["retrievers"]
        retrievers = []
        
        for c_cfg in retrievers_cfg:
            r_type = c_cfg['type']
            r_conf = c_cfg['config']
            
            if r_type not in RETRIEVER_REGISTRY:
                raise ValueError(f"Unknown retriever type: {r_type}")
            
            RetrieverClass = RETRIEVER_REGISTRY[r_type]
            retriever = RetrieverClass.from_config(r_conf)
            retrievers.append(retriever)
            
        fusion_method = config.get("fusion_method", "rrf")
        weights = config.get("weights", None)
        rrf_k = config.get("rrf_k", 60)
        top_k = config.get("top_k", 5)
        
        return cls(
            retrievers=retrievers,
            fusion_method=fusion_method,
            weights=weights,
            rrf_k=rrf_k,
            top_k=top_k
        )
            
    
    async def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k
            
        tasks = [r.retrieve(query) for r in self.retrievers]
        result_lists = await asyncio.gather(*tasks)
        
        fused = self._fuse_results(result_lists)
        
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k] if top_k else fused
    
    async def retrieve_batch(
        self, queries: List[str], top_k: int = None
    ):
        if top_k is None:
            top_k = self.top_k
            
        tasks = [r.retrieve_batch(queries) for r in self.retrievers]
        batch_result_lists = await asyncio.gather(*tasks)
        
        final = []
        Q = len(queries)
        for qi in range(Q):
            per_query_lists = [
                batch_result_lists[ri][qi] for ri in range(len(self.retrievers))
            ]
            fused = self._fuse_results(per_query_lists)
            fused.sort(key=lambda x: x["score"], reverse=True)
            final.append(fused[:top_k] if top_k else fused)
        
        return final
    
    def _fuse_results(self, result_lists: List[List[Dict[str, Any]]]):
        if self.fusion_method == "rrf":
            return self._fusion_rrf(result_lists)
        elif self.fusion_method == "weighted":
            return self._fusion_weighted(result_lists)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
    def _fusion_rrf(self, result_lists: List[List[Dict[str, Any]]]):
        """
        result_lists:
            [
                [ {score, metadata}, ... ],  # retriever1
                [ {score, metadata}, ... ],  # retriever2
            ]

        回傳：
            [
                { "score": fused_score, "metadata": {...} },
                ...
            ]
        """
        score_map = {}
        for results in result_lists:
            for rank, item in enumerate(results):
                doc = item['metadata']
                doc_id = doc.get("id")
                
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                
                if doc_id not in score_map:
                    score_map[doc_id] = {
                        "score": rrf_score,
                        "metadata": doc
                    }
                else:
                    score_map[doc_id]["score"] += rrf_score
        
        return list(score_map.values())
    
    def _fusion_weighted(self, result_lists: List[List[Dict[str, Any]]]):
        score_map = {}
        
        for retriever_idx, results in enumerate(result_lists):
            weight = self.weights[retriever_idx]
            
            scores = [item["score"] for item in results]
            min_s, max_s = min(scores), max(scores)
            denom = (max_s - min_s) or 1.0

            for item in results:
                doc = item["metadata"]
                doc_id = doc.get("id")

                normalized = (item["score"] - min_s) / denom
                fused_score = normalized * weight

                if doc_id not in score_map:
                    score_map[doc_id] = {
                        "score": fused_score,
                        "metadata": doc
                    }
                else:
                    score_map[doc_id]["score"] += fused_score
        
        return list(score_map.values())
    
RETRIEVER_REGISTRY[HybridRetriever.retriever_type] = HybridRetriever