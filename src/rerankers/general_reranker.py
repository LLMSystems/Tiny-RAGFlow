import asyncio
from typing import Any, Dict, List


from ..core.client.embedding_rerank_client import RerankingModel
from .base_reranker import BaseReranker
from .reranker_registry import RERANKER_REGISTRY


class GeneralReranker(BaseReranker):
    reranker_type = "general_reranker"
    def __init__(
        self, 
        model_name: str, 
        config_path: str,
        conbine_metadata_keys: List[str] = None
    ):
        super().__init__()
        self.reranker = RerankingModel(
            reranking_model=model_name,
            config_path=config_path
        )
        self.conbine_metadata_keys = conbine_metadata_keys
        self.logger.info(f"Initialized GeneralReranker with model: {model_name}")
        
    @classmethod
    def from_config(cls, config: Dict):
        return cls(
            model_name=config["model_name"],
            config_path=config["config_path"],
            conbine_metadata_keys=config.get("conbine_metadata_keys", None)
        )
        
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
        texts = [c["metadata"]["text"] for c in candidates]
        
        if self.conbine_metadata_keys:
            for i, c in enumerate(candidates):
                additional_info = []
                for key in self.conbine_metadata_keys:
                    value = self._get_metadata_value(c['metadata'], key)
                    if value:
                        additional_info.append(str(value))
                if additional_info:
                    texts[i] += "\n" + "\n".join(additional_info)
                
        
        new_scores = await self.reranker.rerank_documents(
            documents=texts,
            query=query
        )
        
        reranked_results = []
        for score, c in zip(new_scores, candidates):
            reranked_results.append({
                "score": float(score),
                "metadata": c["metadata"]
            })
            
        reranked_results = sorted(
            reranked_results,
            key=lambda x: x["score"],
            reverse=True
        )

        return reranked_results
    
    async def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:

        tasks = []

        for q, docs in zip(queries, documents_list):
            tasks.append(
                asyncio.create_task(self.rerank(q, docs))
            )

        results = await asyncio.gather(*tasks)

        return results
    
RERANKER_REGISTRY[GeneralReranker.reranker_type] = GeneralReranker