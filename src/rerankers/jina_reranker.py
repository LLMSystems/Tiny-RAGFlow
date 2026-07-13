import asyncio
from typing import Any, Dict, List


from ..core.client.embedding_rerank_client import JinaForRerankingModel
from .base_reranker import BaseReranker
from .reranker_registry import RERANKER_REGISTRY


class JinaReranker(BaseReranker):
    reranker_type = "jina_reranker"
    def __init__(
        self, 
        model_path: str,
        conbine_metadata_keys: List[str] = None
    ):
        super().__init__()
        self.reranker = JinaForRerankingModel(
            model_path=model_path
        )
        self.conbine_metadata_keys = conbine_metadata_keys
        self.logger.info(f"Initialized JinaReranker with model: {model_path}")
        
    @classmethod
    def from_config(cls, config: Dict):
        return cls(
            model_path=config["model_path"],
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
        
        if self.conbine_metadata_keys:
            rerank_documents_list = []
            for d_idx, documents in enumerate(documents_list):
                docs_texts = []
                for c in documents:
                    text = c["metadata"]["text"]
                    additional_info = []
                    for key in self.conbine_metadata_keys:
                        value = self._get_metadata_value(c['metadata'], key)
                        if value:
                            additional_info.append(str(value))
                    if additional_info:
                        text += "\n" + "\n".join(additional_info)
                    docs_texts.append(text)
                rerank_documents_list.append(docs_texts)   
        else:
            rerank_documents_list = [
                [c["metadata"]["text"] for c in documents]
                for documents in documents_list
            ]                 
        
        response = await self.reranker.rerank_documents_batch(
            documents_list=rerank_documents_list,
            query_list=queries
        )

        all_reranked_results = []
        for query_results, candidates in zip(response, documents_list):
            reranked_results = []
            for score, c in zip(query_results, candidates):
                reranked_results.append({
                    "score": float(score),
                    "metadata": c["metadata"]
                })
            reranked_results = sorted(
                reranked_results,
                key=lambda x: x["score"],
                reverse=True
            )
            all_reranked_results.append(reranked_results)   
        return all_reranked_results
    
RERANKER_REGISTRY[JinaReranker.reranker_type] = JinaReranker