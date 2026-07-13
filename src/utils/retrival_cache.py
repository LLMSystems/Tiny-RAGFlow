import os
import json
import hashlib
import asyncio
from datetime import datetime

class RetrievalCacheManager:
    def __init__(self, cache_file="./retriever_cache.json"):
        self.cache_file = cache_file
        self.lock = asyncio.Lock()

        if not os.path.exists(cache_file):
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

        with open(cache_file, "r", encoding="utf-8") as f:
            self.cache = json.load(f)

    def make_key(self, query, top_k, config=None):
        payload = {
            "query": query,
            "top_k": top_k,
            "config": config or {}
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def get(self, key):
        return self.cache.get(key)

    async def set(self, key, config, results):
        self.cache[key] = {
            "config": config,
            "results": results
        }
        await self._save()

    async def _save(self):
        async with self.lock:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)