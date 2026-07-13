from typing import Any, Dict, List

from ..retrievers.qdrant_retriever import QdrantMultivectorRetriever
from .base_reranker import BaseReranker
from .reranker_registry import RERANKER_REGISTRY


class MultivectorReranker(BaseReranker):
    reranker_type = "multivector_reranker"
    def __init__(self, config: Dict):
        super().__init__()
        self.retriever = QdrantMultivectorRetriever.from_config(config)
        self.logger.info(f"Initialized MultivectorReranker with retriever config: {config}")
        
    @classmethod
    def from_config(cls, config: Dict):
        return cls(config=config)
    
    async def rerank(self, query: str, candidates: List[Dict[str, Any]]):
        """
        candidates:
        [
            {
                "score": float,
                "metadata": {"id":..., "text":...}
            },
            ...
        ]

        """
        allowed_ids = [c["metadata"]["id"] for c in candidates]
        reranked_results = await self.retriever.retrieve(query, allowed_ids=allowed_ids)
        
        return reranked_results
    
    async def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        pass
        allowed_ids_list = [
            [doc["metadata"]["id"] for doc in documents]
            for documents in documents_list
        ]
        reranked_results = await self.retriever.retrieve_batch(
            queries,
            allowed_ids_list=allowed_ids_list
        )
        return reranked_results
    
RERANKER_REGISTRY[MultivectorReranker.reranker_type] = MultivectorReranker