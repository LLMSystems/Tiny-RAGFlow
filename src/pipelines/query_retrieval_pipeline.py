import asyncio
from typing import List, Dict, Any

from ..query_enhancers.base_query_enhancer import BaseQueryEnhancer
from ..retrievers.base_retriever import BaseRetriever

class QueryRetrievalPipeline:
    """
    Query → Enhance → Retrieve → Fusion → (Optional Rerank)
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        enhancers: List[BaseQueryEnhancer] = [],
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        top_k: int = 5
    ):
        self.enhancers = enhancers
        self.retriever = retriever
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.top_k = top_k

    async def enhance_query(self, query: str) -> List[str]:
        queries = [query]

        for enhancer in self.enhancers:
            new_list = []
            for q in queries:
                rewrites = await enhancer.enhance(q)
                new_list.extend(rewrites)

            queries = list(set(new_list))

        return queries
    
    async def retrieve(self, query: str, top_k: int = None):
        if top_k is None:
            top_k = self.top_k
        queries = await self.enhance_query(query)

        batch_results = await self.retriever.retrieve_batch(queries, top_k=top_k)

        # 3. 多 query fusion
        fused = self._fuse_batch_results(batch_results)

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]
    
    async def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = None
    ):
        if top_k is None:
            top_k = self.top_k

        enhanced_queries_list = await asyncio.gather(
            *[self.enhance_query(q) for q in queries]
        )

        batch_batch_results = []
        for enhanced_queries in enhanced_queries_list:
            batch_results = await self.retriever.retrieve_batch(enhanced_queries, top_k=top_k)
            batch_batch_results.append(batch_results)

        # 3. Fusion for each original query
        final_results = []
        for batch_results in batch_batch_results:
            fused = self._fuse_batch_results(batch_results)
            fused.sort(key=lambda x: x["score"], reverse=True)
            final_results.append(fused[:top_k])

        return final_results
        
    
    def _fuse_batch_results(self, batch_results: List[List[Dict[str, Any]]]):
        """
        batch_results:
        [
            [ {score, metadata}, ... ],   # from query1
            [ {score, metadata}, ... ],   # from query2
        ]
        """
        score_map = {}

        for result_list in batch_results:
            for rank, item in enumerate(result_list):
                doc = item["metadata"]
                doc_id = doc["id"]

                score = 1.0 / (self.rrf_k + rank + 1)

                if doc_id not in score_map:
                    score_map[doc_id] = {
                        "score": score,
                        "metadata": doc
                    }
                else:
                    score_map[doc_id]["score"] += score

        return list(score_map.values())