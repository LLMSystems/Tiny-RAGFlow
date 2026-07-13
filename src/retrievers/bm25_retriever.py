from typing import Any, Dict, List, Optional, Callable
import os
from ..core.bm25_index import BM25Index
from .base_retriever import BaseRetriever
from .retriever_registry import RETRIEVER_REGISTRY
from ..utils.retrival_cache import RetrievalCacheManager


class BM25Retriever(BaseRetriever):
    retriever_type = "bm25"
    def __init__(
            self, 
            index: BM25Index, 
            top_k: int = 5,
            dedup_key: Optional[str] = None,
            dedup_fn: Optional[Callable] = None,
            config: Dict = None,
            cache_config=None
        ):
        super().__init__(top_k=top_k, dedup_key=dedup_key, dedup_fn=dedup_fn)
        self.index = index
        self.logger.info("BM25Retriever initialized.")
        self.config = config or {}
        self.config['class_name'] = self.__class__.__name__
        self.enable_cache = cache_config.get('enable', True) if cache_config else False
        if self.enable_cache:
            cache_file = cache_config.get('cache_file', './cache/retriever_cache.json')
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            self.cache = RetrievalCacheManager(cache_file=cache_file)
            self.logger.info(f"BM25Retriever cache enabled. Cache file: {cache_file}")
        
        
    @classmethod
    def from_config(cls, config):
        index = BM25Index(config["index_config"], auto_load=True)
        return cls(
            index=index,
            top_k=config.get("top_k", 5),
            dedup_key=config.get("dedup_key", None),
            dedup_fn=config.get("dedup_fn", None),
            config=config,
            cache_config=config.get("cache_config", None)
        )

    async def retrieve(
            self, 
            query: str, 
            top_k: int = None,
            max_retries: int = 3, 
            expansion_factor: int = 2,
            dedup_key: Optional[str] = None, 
            dedup_fn: Optional[Callable] = None
        ) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k
        if dedup_key is None:
            dedup_key = self.dedup_key
        if dedup_fn is None:
            dedup_fn = self.dedup_fn

        if self.enable_cache:
            cache_key = self.cache.make_key(
                query=query,
                top_k=top_k,
                config=self.config
            )

            cached = await self.cache.get(cache_key)
            if cached:
                self.logger.info(f"[BM25Retriever] Cache hit for query='{query}'")
                return cached["results"]

        scores, docs = self.index.search(
            query, 
            top_k,
            max_retries=max_retries,
            expansion_factor=expansion_factor,
            dedup_key=dedup_key,
            dedup_fn=dedup_fn
        )

        result = [
            {"score": float(score), "metadata": doc}
            for score, doc in zip(scores, docs)
        ]

        if self.enable_cache:
            await self.cache.set(
                key=cache_key,
                config=self.config,
                results=result  
            )

        return result

    async def retrieve_batch(
        self, 
        queries: List[str], 
        top_k: int = None,
        max_retries: int = 3, 
        expansion_factor: int = 2,
        dedup_key: Optional[str] = None, 
        dedup_fn: Optional[Callable] = None
    ) -> List[List[Dict[str, Any]]]:
        if top_k is None:
            top_k = self.top_k
        if dedup_key is None:
            dedup_key = self.dedup_key
        if dedup_fn is None:
            dedup_fn = self.dedup_fn

        if not self.enable_cache:
            scores_list, docs_list = self.index.search_batch(
                queries, 
                top_k,
                max_retries=max_retries,
                expansion_factor=expansion_factor,
                dedup_key=dedup_key,
                dedup_fn=dedup_fn
                )

            batch_results = []
            for scores, docs in zip(scores_list, docs_list):
                results = [
                    {
                        "score": float(score),
                        "metadata": doc
                    }
                    for score, doc in zip(scores, docs)
                ]
                batch_results.append(results)

            return batch_results

        cache_keys = [
            self.cache.make_key(
                query=q,
                top_k=top_k,
                config=self.config
            ) for q in queries
        ]

        cached_results = {}
        pending_queries = []
        pending_indices = []

        for idx, key in enumerate(cache_keys):
            cached = await self.cache.get(key)
            if cached:
                cached_results[idx] = cached["results"]
            else:
                pending_queries.append(queries[idx])
                pending_indices.append(idx)

        if len(pending_queries) == 0:
            self.logger.info("All queries hit the cache.")
            return [cached_results[i] for i in range(len(queries))]

        scores_list, docs_list = self.index.search_batch(
                pending_queries, 
                top_k,
                max_retries=max_retries,
                expansion_factor=expansion_factor,
                dedup_key=dedup_key,
                dedup_fn=dedup_fn
                )
        
        for i, global_idx in enumerate(pending_indices):
            scores = scores_list[i]
            docs = docs_list[i]

            result = [
                {"score": float(score), "metadata": doc}
                for score, doc in zip(scores, docs)
            ]

            cached_results[global_idx] = result

            await self.cache.set(
                key=cache_keys[global_idx],
                config=self.config,
                results=result   
            )

        return [cached_results[i] for i in range(len(queries))]


    
RETRIEVER_REGISTRY[BM25Retriever.retriever_type] = BM25Retriever