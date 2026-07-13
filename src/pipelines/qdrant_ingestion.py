import logging
from typing import Dict, List

import numpy as np
from tqdm import tqdm
import re

from ..core.client.embedding_rerank_client import EmbeddingModel, MultiVectorModel
from ..core.qdrant_index import QdrantIndex

class QdrantMultivectorIngestionPipeline:
    def __init__(self, embedding_client: MultiVectorModel, qdrant_index: QdrantIndex):
        self.logger = self._setup_logger()
        self.embedding_client = embedding_client
        self.index = qdrant_index
        
        self.re_emoji = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+")


    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
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
    
    async def ingest_dataset(self, dataset: List[Dict], batch_size: int = 8):
        """
        dataset: List[{
            "id": xx,
            "text": "...",
            ...
        }]
        """
        texts = [self.normalize(item["text"]) for item in dataset]
        metadatas = [{"id": item["id"], **item} for item in dataset]
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Embedding + Ingest"):
            batch_texts = texts[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]

            batch_vectors = await self.embedding_client.embed_documents(batch_texts)

            self.index.add_batch(batch_vectors, batch_meta)

        self.index.save()


class QdrantIngestionPipeline:
    def __init__(self, embedding_client: EmbeddingModel, qdrant_index: QdrantIndex):
        self.logger = self._setup_logger()
        self.embedding_client = embedding_client
        self.index = qdrant_index
        
        self.re_emoji = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+")


    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
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
    
    async def ingest_dataset(self, dataset: List[Dict], batch_size: int = 8):
        """
        dataset: List[{
            "id": xx,
            "text": "...",
            ...
        }]
        """
        texts = [self.normalize(item["text"]) for item in dataset]
        metadatas = [{"id": item["id"], **item} for item in dataset]
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Embedding + Ingest"):
            batch_texts = texts[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]

            batch_vectors = await self.embedding_client.embed_documents(batch_texts)
            batch_vectors = np.array(batch_vectors)

            self.index.add_batch(batch_vectors, batch_meta)

        self.index.save()