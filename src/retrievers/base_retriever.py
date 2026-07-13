import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List


class BaseRetriever(ABC):
    """
    通用 Retriever 基類
    所有 Retriever 都應該繼承這個 class
    """
    def __init__(
            self, 
            top_k: int = 5,
            dedup_key: str = None,
            dedup_fn: Callable = None

        ):
        self.logger = self._setup_logger()
        self.top_k = top_k
        self.dedup_key = dedup_key
        self.dedup_fn = dedup_fn
        
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

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict):
        pass
    
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        傳回格式：
        [
            {
                "score": float,
                "metadata": {...}
            },
            ...
        ]
        """
        pass
    
    @abstractmethod
    async def retrieve_batch(self, queries: List[str], top_k: int = None) -> List[List[Dict[str, Any]]]:
        """
        傳回格式：
        [
            [   # for query1
                {"score":..., "metadata":...},
                ...
            ],
            [   # for query2
                {"score":..., "metadata":...},
                ...
            ]
        ]
        """
        pass
    
    