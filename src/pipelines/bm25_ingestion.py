import logging
import re
from typing import List, Dict

from tqdm import tqdm

from ..core.bm25_index import BM25Index

class BM25IngestionPipeline:
    def __init__(self, bm25_index: BM25Index, auto_fix_id=True):
        self.logger = self._setup_logger()
        self.bm25_index = bm25_index
        self.auto_fix_id = auto_fix_id
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def ingest_dataset(self, dataset: List[Dict], batch_size: int = 32):
        dataset = self._fix_ids_if_needed(dataset)
        
        texts = [item["text"] for item in dataset]
        metadatas = [{"id": item["id"], **item} for item in dataset]
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="BM25 Ingest"):
            batch_texts = texts[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            
            self.bm25_index.add_batch(batch_texts, batch_metas)
            
        self.bm25_index._build_bm25()
        self.bm25_index.save()
        self.logger.info("BM25 ingestion completed and index saved.")
        
    def _fix_ids_if_needed(self, dataset):
        if len(self.bm25_index.metadata) == 0:
            return dataset
        
        existing_ids = [meta["id"] for meta in self.bm25_index.metadata]
        max_id = max(existing_ids)
        expected_id = max_id + 1
        
        fixed_data = []
        
        for item in dataset:
            if item["id"] != expected_id:
                self.logger.warning(f"ID {item['id']} conflicts with existing IDs. Expected ID: {expected_id}.")
                if self.auto_fix_id:
                    self.logger.info(f"Auto-fixing ID {item['id']} to {expected_id}.")
                    item["id"] = expected_id
            fixed_data.append(item)
            expected_id += 1
        
        return fixed_data
