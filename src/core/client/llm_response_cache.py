import asyncio
import hashlib
import json
import os


class LLMResponseCache:
    def __init__(self, cache_file="./llm_cache.json"):
        self.cache_file = cache_file
        self.cache = {}
        self.lock = asyncio.Lock()
        
        if not os.path.exists(cache_file):
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
                
        self._load()
        
    def _load(self):
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self.cache = {}
            
    async def _save(self):
        async with self.lock:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
                
    def make_key(self, model, messages, params):
        raw = json.dumps({
            "model": model,
            "messages": messages,
            "params": params
        }, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    
    async def get(self, key):
        return self.cache.get(key, None)
    
    async def set(self, key, result, model):
        self.cache[key] = {
            "return": result,
            "model": model
        }
        await self._save()
        