import re
import os
from typing import Any, Dict, List, Optional, Callable

import numpy as np

from ..core.client.embedding_rerank_client import EmbeddingModel
from ..core.faiss_index import FaissIndex
from .base_retriever import BaseRetriever
from .retriever_registry import RETRIEVER_REGISTRY
from ..utils.retrival_cache import RetrievalCacheManager

class FaissRetriever(BaseRetriever):
    retriever_type = "faiss"
    def __init__(
        self,
        index: FaissIndex,
        embedder: EmbeddingModel,
        top_k: int = 5,
        dedup_key: Optional[str] = None,
        dedup_fn: Optional[Callable] = None,
        config: Dict = None,
        cache_config=None
    ):
        super().__init__(top_k=top_k, dedup_key=dedup_key, dedup_fn=dedup_fn)
        self.index = index
        self.embedder = embedder
        
        self.re_emoji = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+")
        self.logger.info("FaissRetriever initialized.")
        self.config = config or {}
        self.config['class_name'] = self.__class__.__name__
        self.enable_cache = cache_config.get('enable', True) if cache_config else False
        if self.enable_cache:
            cache_file = cache_config.get('cache_file', './cache/retriever_cache.json')
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            self.cache = RetrievalCacheManager(cache_file=cache_file)
            self.logger.info(f"FaissRetriever cache enabled. Cache file: {cache_file}")
        
    @classmethod
    def from_config(cls, config: Dict):
        index = FaissIndex(config["index_config"], auto_load=True)
        embedder = EmbeddingModel(
            embedding_model=config["embedding_model"],
            config_path=config["model_config_path"]
        )
        return cls(
            index=index,
            embedder=embedder,
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

        query_norm = self.normalize(query)
        
        if self.enable_cache:
            cache_key = self.cache.make_key(
                query=query_norm,
                top_k=top_k,
                config=self.config
            )

            cached = await self.cache.get(cache_key)
            if cached:
                self.logger.info(f"[FaissRetriever] Cache hit for query='{query_norm}'")
                return cached["results"]  

        query_vec = await self.embedder.embed_documents([query_norm])
        query_vec = np.array(query_vec)

        scores, docs = self.index.search(
            query_vec,
            top_k=top_k,
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
        ):
        if top_k is None:
            top_k = self.top_k
        if dedup_key is None:
            dedup_key = self.dedup_key
        if dedup_fn is None:
            dedup_fn = self.dedup_fn

        queries = [self.normalize(q) for q in queries]
        
        if not self.enable_cache:
            query_vecs = np.array(await self.embedder.embed_documents(queries))
            scores_list, docs_list = self.index.search_batch(
                query_vecs,
                top_k=top_k,
                max_retries=max_retries,
                expansion_factor=expansion_factor,
                dedup_key=dedup_key,
                dedup_fn=dedup_fn
            )

            final_results = []
            for scores, docs in zip(scores_list, docs_list):
                final_results.append([
                    {"score": float(score), "metadata": doc}
                    for score, doc in zip(scores, docs)
                ])
            return final_results

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

        pending_vecs = np.array(await self.embedder.embed_documents(pending_queries))

        scores_list, docs_list = self.index.search_batch(
            pending_vecs,
            top_k=top_k,
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


    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        text = self._fullwidth_to_halfwidth(text)  
        text = self.re_emoji.sub("", text)
        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        return text
    
    def _fullwidth_to_halfwidth(self, text):
        """全形 → 半形"""
        res = []
        for char in text:
            code = ord(char)
            if code == 0x3000:       
                code = 0x20
            elif 0xFF01 <= code <= 0xFF5E:  
                code -= 0xFEE0
            res.append(chr(code))
        return "".join(res)
    
RETRIEVER_REGISTRY[FaissRetriever.retriever_type] = FaissRetriever