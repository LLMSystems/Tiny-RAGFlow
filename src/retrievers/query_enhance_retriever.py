import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from ..core.client.llm_client import AsyncLLMChat
from .base_retriever import BaseRetriever
from .retriever_registry import RETRIEVER_REGISTRY

PARAMS = {
    "temperature": 0.2,
    "max_tokens": 2000
}

QUERY_EXPAND_PROMPT_PARAPHRASE_EXPANSION = """
You are an AI language model assistant. Your task is to generate {query_expand_number} \n 
different versions of the given user question to retrieve relevant documents from a vector \n
database. By generating multiple perspectives on the user question, your goal is to help \n
the user overcome some of the limitations of the distance-based similarity search. \n

範例：
原始問題：我想知道信用卡年費怎麼收？
擴展查詢：
{{
  "original_query": "我想知道信用卡年費怎麼收？",
  "expanded_queries": [
    "信用卡的年費是怎麼計算的？",
    "哪些信用卡需要支付年費？費用是多少？",
    "我可以免除信用卡年費嗎？有什麼條件？"
  ]
}}

請根據以下原始問題產生擴展查詢，並以相同格式輸出：

原始問題：{user_query}
擴展查詢：
"""

QUERY_EXPAND_PROMPT_SINGLE_INTENT_DECOMPOSITION = """
你是一個專精於查詢拆解的助理。你的任務是：
- 分析使用者原始問題，判斷是否包含多個意圖。
- 如果包含多個意圖，將其拆解成 {query_expand_number} 個「單一意圖」的子問題。
- 如果原始問題只有單一意圖，則輸出空的 sub_questions 陣列。

--- 規範 ---
1. 每個子問題必須聚焦一個明確的意圖或需求。
2. 子問題必須與原始問題語義相關，不做額外延伸。
3. 保持自然語言表達，避免過度簡化或失去上下文。
4. 如果判斷為單一意圖，sub_questions = []。

--- 範例 ---
原始問題：我想知道信用卡年費怎麼收？還有哪些條件可以免年費？
輸出：
{{
    "original_query": "我想知道信用卡年費怎麼收？還有哪些條件可以免年費？",
    "sub_questions": [
        "信用卡年費的收取方式是什麼？",
        "哪些條件可以免除信用卡年費？"
    ]
}}

原始問題：信用卡年費怎麼收？
輸出：
{{
    "original_query": "信用卡年費怎麼收？",
    "sub_questions": []
}}

請根據以下原始問題產生拆解後的子問題，並以相同格式輸出：
原始問題：{user_query}
輸出：
"""

QUERY_EXPAND_PROMPT_SINGLE_INTENT_DECOMPOSITION_INPORTANCE = """
你是一個專精於查詢拆解與意圖評估的助理。你的任務是：

1. 分析使用者原始問題，判斷是否包含多個意圖。
2. 如果包含多個意圖，將問題拆解成 {query_expand_number} 個「單一意圖」的子問題。
3. 同時，為每個子問題分配一個 0~1 的「重要性比例 importance_ratio」，所有比例加總必須等於 1。
4. 如果原始問題只有單一意圖，則輸出 sub_questions = [] 並輸出 importance_ratio = []。

--- 規範 ---
1. 每個子問題必須聚焦一個明確的意圖或需求。
2. 子問題必須與原始問題語義相關，不做額外延伸。
3. 重要性比例代表該子問題在回答原始問題時的「相對重要性」。
4. 所有 importance_ratio 加總 = 1。
5. 如果判斷為單一意圖：sub_questions = []，importance_ratio = []。

--- 範例 1 ---
原始問題：我想知道信用卡年費怎麼收？還有哪些條件可以免年費？
輸出：
{{
    "original_query": "我想知道信用卡年費怎麼收？還有哪些條件可以免年費？",
    "sub_questions": [
        "信用卡年費的收取方式是什麼？",
        "哪些條件可以免除信用卡年費？"
    ],
    "importance_ratio": [0.5, 0.5]
}}

--- 範例 2 ---
原始問題：信用卡年費怎麼收？
輸出：
{{
    "original_query": "信用卡年費怎麼收？",
    "sub_questions": [],
    "importance_ratio": []
}}

請根據以下原始問題產生拆解後的子問題及其重要性比例，並以相同 JSON 格式輸出：
原始問題：{user_query}
輸出：
"""




QUERY_EXPAND_PROMPT_HYDE_EXPANSION = """
You are an AI assistant specialized in Hypothetical Document Expansion (HyDE).  
Your task is to generate a short hypothetical passage that could reasonably 
appear in a knowledge base and would answer the user's question. The passage 
should sound factual, neutral, and informative, but it does not need to be 
correct — it only needs to be realistic enough to help retrieval.

After generating the passage, rewrite the content into {query_expand_number} 
search queries that capture the key concepts from the hypothetical passage.

--- Guidelines for HyDE Passage ---
1. The passage MUST look like it was extracted from an article, FAQ, financial 
   guide, or knowledge base.
2. The passage MUST be concise (3–5 sentences).
--- Guidelines for Expanded Queries ---
1. Each query must be derived from the hypothetical passage.

範例：
原始問題：我想知道信用卡年費怎麼收？
假設性文件擴展：
{{
    "original_query": "我想知道信用卡年費怎麼收？",
    "hypothetical_passage": "大多數信用卡會依卡片等級收取固定年費，例如普卡年費較低，而高端卡年費較高。銀行通常會在每年卡片到期月自動扣收，並提供首年免年費或常用交易達標後退費的制度。持卡人也可透過綁定電子帳單或設定自動付款來確保年費扣款流程順利。",
    "expanded_queries": [
        "信用卡年費是依照卡片等級如何計算？",
        "信用卡年費通常在什麼時間點扣款？",
        "哪些條件可以取得信用卡的免年費優惠？"
    ]
}}

請根據以下原始問題產生假設性文件及擴展查詢，並以相同格式輸出：
原始問題：{user_query}
假設性文件擴展：
"""




class QueryEnhanceRetriever(BaseRetriever):
    retriever_type = "queryenhance"
    def __init__(self, 
                 retrievers,
                 llmchater,
                 fusion_method: str = "rrf",
                 top_k: int = 5,
                 rrf_k: int = 60,
                 weights: Optional[List[float]] = None,
                 query_extension_config: Optional[Dict] = None
                ):
        super().__init__(top_k=top_k)
        self.retrievers = retrievers
        if len(retrievers) < 2:
            self.logger.warning("QueryEnhanceRetriever initialized with less than two retrievers. fusion may be unnecessary.")
        self.llmchater = llmchater
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.weights = weights
        self.query_extension_config = query_extension_config
        self.query_expand_number = query_extension_config.get("query_expand_number", 3) if query_extension_config else 3
        self.query_expansion_method = query_extension_config.get("method", "paraphrase") if query_extension_config else "paraphrase"
        self.query_alpha = query_extension_config.get("alpha", 0.3) if query_extension_config else 0.3
        self.query_fusion_method = query_extension_config.get("fusion_method", "round_robin") if query_extension_config else "rrf"
        self.logger.info(f"Query expansion method: {self.query_expansion_method}, number: {self.query_expand_number}, alpha: {self.query_alpha}, fusion: {self.query_fusion_method}")
        
        if self.fusion_method == "weighted":
            if weights is None or len(weights) != len(retrievers):
                raise ValueError("Weights must be provided and match the number of retrievers for weighted fusion.")
        self.logger.info("QueryEnhanceRetriever initialized.")
        
    @classmethod
    def from_config(cls, config:Dict):
        """
        expansion_config = {
        "retrievers": [
                {
                    "type": "faiss",
                    "config": {
                        "index_config": "./config/faiss.yaml",
                        "embedding_model": "m3e-base",
                        "model_config_path": "./config/models.yaml",
                        "top_k": 3
                    }
                },
                {
                    "type": "bm25",
                    "config": {
                        "index_config": "./config/bm25.yaml",
                        "top_k": 3
                    }
                }
            ],
            "fusion_method": "rrf",
            "rrf_k": 60,
            "top_k": 5,
            "query_extension_config": {
                "method": "hyde"  # or "paraphrase" or "sub_question",
                "fusion_method": "round_robin", # or "rrf"
                "query_expand_number": 3,
                "alpha": 0.3
            },
            
            "llm_model": "Qwen2.5-32B-Instruct-GPTQ-Int4",
            "model_config_path": "./config/models.yaml",
            "cache_config": {
                    "enable": True,
                    "cache_file": './cache/query_enhance/retriever_cache.json'
                }
        }
        """
        llmchater = AsyncLLMChat(
            model=config["llm_model"],
            config_path=config["model_config_path"],
            cache_config=config.get("cache_config", None)
            )
        
        retrievers_cfg = config["retrievers"]
        retrievers = []
        for c_cfg in retrievers_cfg:
            r_type = c_cfg['type']
            r_conf = c_cfg['config']
            
            if r_type not in RETRIEVER_REGISTRY:
                raise ValueError(f"Unknown retriever type: {r_type}")
            
            RetrieverClass = RETRIEVER_REGISTRY[r_type]
            retriever = RetrieverClass.from_config(r_conf)
            retrievers.append(retriever)

        return cls(
                retrievers=retrievers,
                llmchater=llmchater,
                fusion_method=config.get("fusion_method", "rrf"),
                rrf_k=config.get("rrf_k", 60),
                weights=config.get("weights", None),
                top_k = config.get("top_k", 5),
                query_extension_config = config.get("query_extension_config", None)
                )
    
    async def retrieve(self, query: str, top_k: int = None, alpha: float = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k
        if alpha is None:
            alpha = self.query_alpha

        expansions, importance_ratios = await self._expand_queries(query)
        all_queries = [query] + expansions
        importance_ratios = self.add_origin_query_ratio(importance_ratios, alpha=alpha)
        num_qi = len(all_queries)

        if len(importance_ratios) != num_qi:
            self.logger.warning(
                f"importance_ratios length {len(importance_ratios)} != num_qi {num_qi}, "
                "fall back to uniform."
            )
            importance_ratios = [1.0 / num_qi] * num_qi

        tasks = [r.retrieve_batch(all_queries) for r in self.retrievers]
        results_per_retriever = await asyncio.gather(*tasks)
        
        if self.query_fusion_method == "rrf":
            final_results = []
            for res in results_per_retriever:
                fused = self._fuse_results(res)
                fused.sort(key=lambda x: x["score"], reverse=True)
                final_results.append(fused)

            if len(final_results) == 1:
                self.logger.info("Only one retriever used; skipping fusion.")
                return final_results[0][:top_k]

            final_fused = self._fuse_results(final_results)
            final_fused.sort(key=lambda x: x["score"], reverse=True)
            return final_fused[:top_k]
        elif self.query_fusion_method == "round_robin":
            self.logger.info("Using round robin fusion for expanded queries.")
            final_results = self._fuse_multi_retriever_results(
                results_lists=results_per_retriever,
                importance_ratios=importance_ratios,
                top_k=top_k
            )
            return final_results
        else:
            raise ValueError(f"Unknown query fusion method: {self.query_fusion_method}")

    async def retrieve_batch(self, queries: List[str], top_k: int = None, alpha: float = None) -> List[List[Dict[str, Any]]]:
        if top_k is None:
            top_k = self.top_k
        if alpha is None:
            alpha = self.query_alpha

        expand_tasks = [self._expand_queries(q) for q in queries]
        results = await asyncio.gather(*expand_tasks)
        
        expanded_list = []
        importance_ratios_list = []
        
        for i, (expanded_qs, ratios) in enumerate(results):
            expanded_qs = [queries[i]] + list(expanded_qs)  # prepend original query
            ratios = self.add_origin_query_ratio(list(ratios), alpha=alpha)

            expanded_list.append(expanded_qs)
            importance_ratios_list.append(ratios)
        
        flat_queries = [q for expanded in expanded_list for q in expanded]

        tasks = [r.retrieve_batch(flat_queries) for r in self.retrievers]
        results_per_retriever = await asyncio.gather(*tasks)

        split_results = []
        for retr_res in results_per_retriever:
            per_query = []
            idx = 0
            for expanded in expanded_list:
                L = len(expanded)
                per_query.append(retr_res[idx:idx+L])
                idx += L
            split_results.append(per_query)
            
        if self.query_fusion_method == "rrf":
            for r in range(len(split_results)):
                for q in range(len(split_results[r])):
                    fused = self._fuse_results(split_results[r][q])
                    fused.sort(key=lambda x: x["score"], reverse=True)
                    split_results[r][q] = fused

            if len(self.retrievers) == 1:
                self.logger.info("Only one retriever used; skipping fusion.")
                return [
                    split_results[0][q][:top_k]
                    for q in range(len(queries))
                ]

            final_results = []
            num_queries = len(queries)
            for q in range(num_queries):
                per_query_across_retrievers = [
                    split_results[r][q]
                    for r in range(len(self.retrievers))
                ]
                fused = self._fuse_results(per_query_across_retrievers)
                fused.sort(key=lambda x: x["score"], reverse=True)
                final_results.append(fused[:top_k])

            return final_results
        elif self.query_fusion_method == "round_robin":
            self.logger.info("Using round robin fusion for expanded queries.")

            fused_results = []

            num_retrievers = len(self.retrievers)
            num_queries = len(queries)

            for qi in range(num_queries):
                # results_lists = list of [retriever][qi] docs
                results_lists = [split_results[r][qi] for r in range(num_retrievers)]

                fused = self._fuse_multi_retriever_results(
                    results_lists=results_lists,
                    importance_ratios=importance_ratios_list[qi],
                    top_k=top_k
                )
                fused_results.append(fused)

            return fused_results
        else:
            raise ValueError(f"Unknown query fusion method: {self.query_fusion_method}")
    
    def _fuse_multi_retriever_results(
        self,
        results_lists: List[List[List[Dict[str, Any]]]],
        importance_ratios: List[float],
        top_k: int
    ):
        num_retrievers = len(results_lists)
        num_qi = len(results_lists[0])
        
        if num_retrievers == 1:
            self.logger.info("Only one retriever used; skip retriever-level fusion.")
            results = results_lists[0] # first retriever
            selected_per_qi = self.distrite_results_by_quota(
                results=results,
                importance_ratios=importance_ratios,
                top_k=top_k,
            )
            fused = self.round_robin_sort(selected_per_qi, top_k=top_k)
            return fused[:top_k]
        # 多個 retriever 的結果融合
        qi_level_results: List[List[List[Dict[str, Any]]]] = []
        for qi in range(num_qi):
            per_qi_across_retrievers = [
                results_lists[r][qi]
                for r in range(num_retrievers)
            ]
            qi_level_results.append(per_qi_across_retrievers)

        fused_per_qi: List[List[Dict[str, Any]]] = []
        for qi_results in qi_level_results:
            fused = self._fuse_results(qi_results)  
            fused.sort(key=lambda x: x["score"], reverse=True)
            fused_per_qi.append(fused)  # [qi][docs]

        selected_per_qi = self.distrite_results_by_quota(
            results=fused_per_qi,         
            importance_ratios=importance_ratios,
            top_k=top_k,
        )

        final_fused = self.round_robin_sort(selected_per_qi, top_k=top_k)
        return final_fused[:top_k]
        
            
    def distrite_results_by_quota(self, results: List[List[Dict[str, Any]]], importance_ratios: List[float], top_k: int):
        """
        results: 
            [
                [ {score, metadata}, ... ],  # query 1
                [ {score, metadata}, ... ],  # query 2
                ...
            ]
        importance_ratios: [r1, r2, ...]  # sum = 1
        """
        num_queries = len(results)
        if num_queries == 0 or top_k <= 0:
            return [[] for _ in range(num_queries)]
        if num_queries != len(importance_ratios):
            self.logger.warning("Number of results and importance ratios do not match.")
            importance_ratios = [1.0 / num_queries] * num_queries
        
        total_ratio = sum(importance_ratios)
        if total_ratio <= 0:
            importance_ratios = [1.0 / num_queries] * num_queries
        else:
            importance_ratios = [r / total_ratio for r in importance_ratios]
        
        raw_quotas = [r * top_k for r in importance_ratios]
        quotas = [int(q) for q in raw_quotas]
        
        deficit = top_k - sum(quotas)
        if deficit > 0:
            frac = [(i, raw_quotas[i] - quotas[i]) for i in range(num_queries)]
            frac.sort(key=lambda x: x[1], reverse=True)
            for i, _ in frac:
                if deficit == 0:
                    break
                quotas[i] += 1
                deficit -= 1

        overflow = sum(quotas) - top_k
        if overflow > 0:
            sorted_idx = sorted(range(num_queries), key=lambda i: quotas[i], reverse=True)
            for idx in sorted_idx:
                if overflow == 0:
                    break
                if quotas[idx] > 0:
                    quotas[idx] -= 1
                    overflow -= 1

        selected_per_qi: List[List[Dict[str, Any]]] = []
        for q_idx, q_quota in enumerate(quotas):
            docs = results[q_idx]
            docs_sorted = sorted(docs, key=lambda x: x["score"], reverse=True)
            selected_per_qi.append(docs_sorted[:q_quota])

        return selected_per_qi
    
    def round_robin_sort(self, results_all_qi: List[List[Dict[str, Any]]], top_k: int):
        rr: List[Dict[str, Any]] = []
        layer = 0
        while len(rr) < top_k:
            any_added = False

            for qi_results in results_all_qi:   
                if layer < len(qi_results):
                    rr.append(qi_results[layer])
                    any_added = True
                    if len(rr) == top_k:
                        break

            if not any_added:
                break
            layer += 1
        return rr

    def add_origin_query_ratio(self, importance_ratios, alpha=0.3):
        """
        importance_ratios: [r1, r2, ...]  # sum = 1
        alpha: Q 的固定比重（建議 0.2 ~ 0.4）
        """
        scale = 1 - alpha
        adjusted = [r * scale for r in importance_ratios]
        final = [alpha] + adjusted
        return final

    async def _expand_queries(self, query: str) -> List[str]:
        key1 = None
        if self.query_expansion_method == "paraphrase":
            prompt = QUERY_EXPAND_PROMPT_PARAPHRASE_EXPANSION.format(user_query=query, query_expand_number=self.query_expand_number)
            key0 = "expanded_queries"
        elif self.query_expansion_method == "sub_question":
            key0 = "sub_questions"
            if self.query_fusion_method == "rrf":
                prompt = QUERY_EXPAND_PROMPT_SINGLE_INTENT_DECOMPOSITION.format(user_query=query, query_expand_number=self.query_expand_number)
            else:
                prompt = QUERY_EXPAND_PROMPT_SINGLE_INTENT_DECOMPOSITION_INPORTANCE.format(user_query=query, query_expand_number=self.query_expand_number)
                key1 = "importance_ratio"
        elif self.query_expansion_method == "hyde":
            prompt = QUERY_EXPAND_PROMPT_HYDE_EXPANSION.format(user_query=query, query_expand_number=self.query_expand_number)
            key0 = "expanded_queries"
        else:
            raise ValueError(f"Unknown query expansion method: {self.query_expansion_method}")
        
        raw_response, _ = await self.llmchater.chat(prompt, params=PARAMS)
        raw_response = self._normalize_json_response(raw_response)
        response = json.loads(raw_response)[key0]
        if key1:
            importance_ratios = json.loads(raw_response)[key1]
            if len(response) != len(importance_ratios):
                self.logger.warning("Length of expanded queries and importance ratios do not match.")
                importance_ratios = [1.0 / len(response)] * len(response)
        else:
            # 處理response = [] 的情況，len=0時避免除以0錯誤
            if len(response) == 0:
                importance_ratios = []
            else:
                importance_ratios = [1.0 / len(response)] * len(response)
        return response, importance_ratios

    def _normalize_json_response(self, response: str) -> str:
        if not response:
            return "{}"
        
        response_clean = response.strip()
        
        json_block_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'`json\s*(.*?)\s*`',
            r'`\s*(.*?)\s*`'
        ]
        
        for pattern in json_block_patterns:
            match = re.search(pattern, response_clean, re.DOTALL)
            if match:
                response_clean = match.group(1).strip()
                break
        
        start_idx = response_clean.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(response_clean[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            
            if brace_count == 0:
                response_clean = response_clean[start_idx:end_idx + 1]
        
        response_clean = re.sub(r"'([^']*)':", r'"\1":', response_clean)
        response_clean = re.sub(r":\s*'([^']*)'", r': "\1"', response_clean)
        response_clean = re.sub(r',\s*}', '}', response_clean)
        response_clean = re.sub(r',\s*]', ']', response_clean)
        
        return response_clean
    
    def _fuse_results(self, result_lists: List[List[Dict[str, Any]]]):
        if self.fusion_method == "rrf":
            return self._fusion_rrf(result_lists)
        elif self.fusion_method == "weighted":
            return self._fusion_weighted(result_lists)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
    def _fusion_rrf(self, result_lists: List[List[Dict[str, Any]]]):
        """
        result_lists:
            [
                [ {score, metadata}, ... ],  # retriever1
                [ {score, metadata}, ... ],  # retriever2
            ]

        回傳：
            [
                { "score": fused_score, "metadata": {...} },
                ...
            ]
        """
        score_map = {}
        for results in result_lists:
            for rank, item in enumerate(results):
                doc = item['metadata']
                doc_id = doc.get("id")
                
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                
                if doc_id not in score_map:
                    score_map[doc_id] = {
                        "score": rrf_score,
                        "metadata": doc
                    }
                else:
                    score_map[doc_id]["score"] += rrf_score
        
        return list(score_map.values())
    
    def _fusion_weighted(self, result_lists: List[List[Dict[str, Any]]]):
        score_map = {}
        
        for retriever_idx, results in enumerate(result_lists):
            weight = self.weights[retriever_idx]
            
            scores = [item["score"] for item in results]
            min_s, max_s = min(scores), max(scores)
            denom = (max_s - min_s) or 1.0

            for item in results:
                doc = item["metadata"]
                doc_id = doc.get("id")

                normalized = (item["score"] - min_s) / denom
                fused_score = normalized * weight

                if doc_id not in score_map:
                    score_map[doc_id] = {
                        "score": fused_score,
                        "metadata": doc
                    }
                else:
                    score_map[doc_id]["score"] += fused_score
        
        return list(score_map.values())
    
RETRIEVER_REGISTRY[QueryEnhanceRetriever.retriever_type] = QueryEnhanceRetriever