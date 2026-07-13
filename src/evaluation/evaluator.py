from typing import Any, Dict, List

from tqdm import tqdm

from ..retrievers.base_retriever import BaseRetriever
from .dataset_loader import EvaluationDataset, MultiIntentEvaluationDataset
from .metrics import (average, hit_rate_at_k, mean_reciprocal_rank, ndcg_at_k,
                      precision_at_k, recall_at_k, hit_rate_at_k_multi, mean_reciprocal_rank_multi, ndcg_at_k_multi, precision_at_k_multi, recall_at_k_multi)

class MultiIntentRetrieverEvaluator:
    def __init__(self, retriever: BaseRetriever, eval_dataset: MultiIntentEvaluationDataset, task: str = None):
        self.retriever = retriever
        self.eval_dataset = eval_dataset
        self.task = task

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'
    
    async def evaluate(
        self,
        top_k: List[int] = [5, 7, 10],
        batch_size: int = 8,
        task: str = None
    ) -> Dict[str, Any]:
        if task is None and self.task is not None:
            task = self.task

        max_k = max(top_k)

        hit_rates = {k: [] for k in top_k}
        recalls = {k: [] for k in top_k}
        precisions = {k: [] for k in top_k}
        ndcgs = {k: [] for k in top_k}
        mrr_scores = []

        by_intent_count = {}

        total_samples = len(self.eval_dataset)
        
        details = []

        for start in tqdm(range(0, total_samples, batch_size), desc="Evaluating Retriever"):
            end = min(start + batch_size, total_samples)
            batch_samples = self.eval_dataset._samples[start:end]

            batch_queries = [s.query for s in batch_samples]
            if task is not None:
                batch_queries = [self.get_detailed_instruct(task, q) for q in batch_queries]
            batch_ground_truth_list = [s.ground_truth_ids_list for s in batch_samples]

            batch_results = await self.retriever.retrieve_batch(batch_queries, top_k=max_k)

            for sample, results, gt_ids_list in zip(batch_samples, batch_results, batch_ground_truth_list):
                details.append({
                    "query": sample.query,
                    "ground_truth_ids_list": gt_ids_list,
                    "source_queries": sample.source_queries,
                    "intent_count": sample.intent_count,
                    "results": results,
                })
                
                # metadata
                if sample.metadata is not None:
                    details[-1]["sample_metadata"] = sample.metadata

                """
                multi intent result data structure
                results: 
                [   
                    {
                        "score": float,
                        "metadata": {
                            "id": int,
                            ...
                        }
                    },
                    {
                        "score": float,
                        "metadata": {
                            "id": int,
                            ...
                        }
                    }
                ]
                gt_ids_list:
                all intent ground truth ids list
                [
                    [int, int, ...],  # intent 1 ground truth ids
                    [int, int, ...],  # intent 2 ground truth ids
                    ...
                ]
                """

                # group by intent
                intent_count = sample.intent_count
                if intent_count not in by_intent_count:
                    by_intent_count[intent_count] = {
                        "hit_rates": {k: [] for k in top_k},
                        "recalls": {k: [] for k in top_k},
                        "precisions": {k: [] for k in top_k},
                        "ndcgs": {k: [] for k in top_k},
                        "mrr_scores": []
                    }

                for k in top_k:
                    hit_rates[k].append(hit_rate_at_k_multi(results, gt_ids_list, k))
                    recalls[k].append(recall_at_k_multi(results, gt_ids_list, k))
                    precisions[k].append(precision_at_k_multi(results, gt_ids_list, k))
                    ndcgs[k].append(ndcg_at_k_multi(results, gt_ids_list, k))

                    by_intent_count[intent_count]["hit_rates"][k].append(hit_rate_at_k_multi(results, gt_ids_list, k))
                    by_intent_count[intent_count]["recalls"][k].append(recall_at_k_multi(results, gt_ids_list, k))
                    by_intent_count[intent_count]["precisions"][k].append(precision_at_k_multi(results, gt_ids_list, k))
                    by_intent_count[intent_count]["ndcgs"][k].append(ndcg_at_k_multi(results, gt_ids_list, k))

                mrr_score = mean_reciprocal_rank_multi(results, gt_ids_list)
                mrr_scores.append(mrr_score)

                by_intent_count[intent_count]["mrr_scores"].append(mrr_score)

        summary = {"by_k": {}, "by_intent_count": {}}

        for k in top_k:
            summary["by_k"][k] = {
                "HitRate": average(hit_rates[k]),
                "Recall": average(recalls[k]),
                "Precision": average(precisions[k]),
                "NDCG": average(ndcgs[k]),
            }

        summary["MRR"] = average(mrr_scores)

        for intent_count, metrics in by_intent_count.items():
            summary["by_intent_count"][intent_count] = {"by_k": {}}
            for k in top_k:
                summary["by_intent_count"][intent_count]["by_k"][k] = {
                    "HitRate": average(metrics["hit_rates"][k]),
                    "Recall": average(metrics["recalls"][k]),
                    "Precision": average(metrics["precisions"][k]),
                    "NDCG": average(metrics["ndcgs"][k]),
                }
            summary["by_intent_count"][intent_count]["MRR"] = average(metrics["mrr_scores"])

        return {
            "summary": summary,
            "details": details
        }

class RetrieverEvaluator:
    """
    Standard evaluation module for retrievers.

    Features:
    ---------
    - async evaluation loop
    - supports multiple top-k metrics
    - compatible with retriever.retrieve(query)
    - expects result format:
        [
          {
            "score": 0.98,
            "metadata": {"id": 12, "text": "..."}
          },
          ...
        ]
    """
    def __init__(self, retriever: BaseRetriever, eval_dataset: EvaluationDataset, task: str = None):
        self.retriever = retriever
        self.eval_dataset = eval_dataset
        self.task = task

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'
        
    async def evaluate(
        self,
        top_k: List[int] = [5],
        batch_size: int = 8,
        task: str = None
    ) -> Dict[str, Any]:
        if task is None and self.task is not None:
            task = self.task

        max_k = max(top_k)

        hit_rates = {k: [] for k in top_k}
        recalls = {k: [] for k in top_k}
        precisions = {k: [] for k in top_k}
        ndcgs = {k: [] for k in top_k}
        mrr_scores = []

        total_samples = len(self.eval_dataset)
        
        details = []

        for start in tqdm(range(0, total_samples, batch_size), desc="Evaluating Retriever"):
            end = min(start + batch_size, total_samples)
            batch_samples = self.eval_dataset._samples[start:end]

            batch_queries = [s.query for s in batch_samples]
            if task is not None:
                batch_queries = [self.get_detailed_instruct(task, q) for q in batch_queries]
            batch_ground_truth = [s.ground_truth_ids for s in batch_samples]

            batch_results = await self.retriever.retrieve_batch(batch_queries, top_k=max_k)
            
            for sample, results, gt_ids in zip(batch_samples, batch_results, batch_ground_truth):
                details.append({
                    "query": sample.query,
                    "ground_truth_ids": gt_ids,
                    "results": results,
                })
                
                # metadata
                if sample.metadata is not None:
                    details[-1]["sample_metadata"] = sample.metadata

                for k in top_k:
                    hit_rates[k].append(hit_rate_at_k(results, gt_ids, k))
                    recalls[k].append(recall_at_k(results, gt_ids, k))
                    precisions[k].append(precision_at_k(results, gt_ids, k))
                    ndcgs[k].append(ndcg_at_k(results, gt_ids, k))

                mrr_scores.append(mean_reciprocal_rank(results, gt_ids))

        summary = {}

        summary = {"by_k": {}}

        for k in top_k:
            summary["by_k"][k] = {
                "HitRate": average(hit_rates[k]),
                "Recall": average(recalls[k]),
                "Precision": average(precisions[k]),
                "NDCG": average(ndcgs[k]),
            }

        summary["MRR"] = average(mrr_scores)

        return {
            "summary": summary,
            "details": details
        }