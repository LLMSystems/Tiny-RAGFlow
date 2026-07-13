import math
from typing import Dict, List


def extract_ids(results: List[Dict]) -> List[int]:
    """
    Extract document IDs from retriever results.

    Expected result format:
        [
          {
            "score": float,
            "metadata": {"id": int, "text": str}
          },
          ...
        ]

    Returns:
        [id1, id2, id3 ...] in ranked order
    """
    return [item["metadata"]["id"] for item in results]

def ensure_weights(m, weights=None):
    if weights is None:
        return [1.0/m] * m
    assert len(weights) == m, "Length of intent_weights must match number of intents"
    s = sum(weights)
    return [w/s for w in weights]

def hit_rate_at_k_multi(result: List[Dict], ground_truth_ids_list: List[List[int]], k: int, intent_weights=None) -> float:
    ranked_ids = extract_ids(result)
    m = len(ground_truth_ids_list)
    w = ensure_weights(m, intent_weights)
    top_k_ids = set(ranked_ids[:k])
    hit = [(1 if top_k_ids.intersection(set(gt_ids)) else 0) for gt_ids in ground_truth_ids_list]
    return sum([hi * wi for hi, wi in zip(hit, w)])


def recall_at_k_multi(result: List[Dict], ground_truth_ids_list: List[List[int]], k: int, intent_weights=None) -> float:
    ranked_ids = extract_ids(result)
    m = len(ground_truth_ids_list)
    w = ensure_weights(m, intent_weights)
    top_k_ids = set(ranked_ids[:k])
    recalls = []
    for gt_ids in ground_truth_ids_list:
        if len(gt_ids) == 0:
            recalls.append(0.0)
        else:
            hit_count = sum(1 for gt in gt_ids if gt in top_k_ids)
            recalls.append(hit_count / len(gt_ids))
    return sum([ri * wi for ri, wi in zip(recalls, w)])

def precision_at_k_multi(result: List[Dict], ground_truth_ids_list: List[List[int]], k: int, intent_weights=None) -> float:
    ranked_ids = extract_ids(result)
    m = len(ground_truth_ids_list)
    w = ensure_weights(m, intent_weights)
    top_k_ids = set(ranked_ids[:k])
    precisions = []
    for gt_ids in ground_truth_ids_list:
        hit_count = sum(1 for id_ in top_k_ids if id_ in gt_ids)
        precisions.append(hit_count / k if k > 0 else 0.0)
    return sum([pi * wi for pi, wi in zip(precisions, w)])

def mean_reciprocal_rank_multi(result: List[Dict], ground_truth_ids_list: List[List[int]], intent_weights=None) -> float:
    ranked_ids = extract_ids(result)
    m = len(ground_truth_ids_list)
    w = ensure_weights(m, intent_weights)
    mrrs = []
    for gt_ids in ground_truth_ids_list:
        rr = 0.0
        for idx, doc_id in enumerate(ranked_ids):
            if doc_id in gt_ids:
                rr = 1.0 / (idx + 1)
                break
        mrrs.append(rr)
    return sum([mrri * wi for mrri, wi in zip(mrrs, w)])

def ndcg_at_k_multi(result: List[Dict], ground_truth_ids_list: List[List[int]], k: int, intent_weights=None) -> float:
    ranked_ids = extract_ids(result)
    m = len(ground_truth_ids_list)
    w = ensure_weights(m, intent_weights)
    ndcgs = []
    for gt_ids in ground_truth_ids_list:
        # DCG calculation
        dcg = 0.0
        for idx, doc_id in enumerate(ranked_ids[:k]):
            if doc_id in gt_ids:
                dcg += 1.0 / math.log2(idx + 2)

        # IDCG calculation
        ideal_hits = min(len(gt_ids), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    return sum([ndcgi * wi for ndcgi, wi in zip(ndcgs, w)])


def hit_rate_at_k(results: List[Dict], ground_truth_ids: List[int], k: int) -> float:
    """
    Hit-Rate@K: whether at least one ground truth ID
    appears in the top-k results.
    """
    top_k_ids = extract_ids(results[:k])
    return 1.0 if any(gt in top_k_ids for gt in ground_truth_ids) else 0.0


def recall_at_k(results: List[Dict], ground_truth_ids: List[int], k: int) -> float:
    """
    Recall@K: (# of ground truths retrieved in top-k) / (# of total ground truths)
    """
    top_k_ids = extract_ids(results[:k])
    hit_count = sum(1 for gt in ground_truth_ids if gt in top_k_ids)
    return hit_count / len(ground_truth_ids) if ground_truth_ids else 0.0


def precision_at_k(results: List[Dict], ground_truth_ids: List[int], k: int) -> float:
    """
    Precision@K: (# of retrieved relevant docs in top-k) / k
    """
    top_k_ids = extract_ids(results[:k])
    hit_count = sum(1 for id_ in top_k_ids if id_ in ground_truth_ids)
    return hit_count / k if k > 0 else 0.0


def mean_reciprocal_rank(results: List[Dict], ground_truth_ids: List[int]) -> float:
    """
    MRR: reciprocal rank of the first relevant document.
    """
    ranked_ids = extract_ids(results)
    for idx, doc_id in enumerate(ranked_ids):
        if doc_id in ground_truth_ids:
            return 1.0 / (idx + 1)
    return 0.0


def ndcg_at_k(results: List[Dict], ground_truth_ids: List[int], k: int) -> float:
    """
    NDCG@K based on binary relevance (1 if in ground truth else 0).
    """
    ranked_ids = extract_ids(results[:k])

    dcg = 0.0
    for idx, doc_id in enumerate(ranked_ids):
        if doc_id in ground_truth_ids:
            dcg += 1.0 / math.log2(idx + 2)

    ideal_hits = min(len(ground_truth_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
