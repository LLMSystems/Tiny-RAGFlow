import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseReranker(ABC):
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _get_metadata_value(self, meta: Dict, key: str):
        parts = key.split(".")
        v = meta
        try:
            for p in parts:
                v = v[p]
            return v
        except Exception:
            return None
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict):
        pass

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str]
    ) -> List[Dict[str, Any]]:
        """
        documents 格式來自 retriever：
        [
            doc1,
            doc2,
            ...
        ]

        reranker 需回傳同樣資料，但重新排序、重新打分
        """
        pass
    
    @abstractmethod
    async def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        queries = ["q1", "q2", ...]
        documents_list = [
            [doc1, doc2, ...],     # docs 對應 q1
            [doc3, doc4, ...],     # docs 對應 q2
            ...
        ]

        回傳：
        [
            [  # for q1
                {"score": ..., "metadata": ...},
                ...
            ],
            [  # for q2
                {"score": ..., "metadata": ...},
                ...
            ]
        ]
        """
        pass