from typing import Any, Dict, List
import asyncio

from .general_reranker import GeneralReranker
from .reranker_registry import RERANKER_REGISTRY

PREFIX = ("<|im_start|>system\n"
          "Judge whether the Document meets the requirements based on the "
          "Query and the Instruct provided. Note that the answer can only be "
          "\"yes\" or \"no\".<|im_end|>\n"
          "<|im_start|>user\n")

SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

INSTRUCTION = "根據用戶問題，找出最相關的文檔"

class QwenReranker(GeneralReranker):
    reranker_type = "qwen_reranker"
    
    def formate_query(self, query: str) -> str:
        return f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {query}"
    
    def formate_documents(self, documents: List[str]) -> List[str]:
        return [f"<Document>: {doc}{SUFFIX}" for doc in documents]
    
    async def rerank(self, query: str, candidates: List[Dict[str, Any]]):
        query = self.formate_query(query)
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
                    
        texts = self.formate_documents(texts)
                
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
    
RERANKER_REGISTRY[QwenReranker.reranker_type] = QwenReranker