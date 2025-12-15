import logging
import asyncio
import yaml
from openai import AsyncOpenAI
import httpx


class BaseModel:
    def __init__(self, model_type, model_name, config_path='./configs/models.yaml'):
        self.config = self.load_config(config_path)
        self.model_config = self.config[model_type][model_name]
        self.model = self.model_config['model']
        self.logger = self._setup_logger()
        self.local_api_key = self.model_config['local_api_key']
        self.local_base_url = self.model_config['local_base_url']
        self.logger.info(f'[{model_type}] Initializing Model: {model_name}')
        self.client = AsyncOpenAI(api_key=self.local_api_key, base_url=self.local_base_url)
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('EmbeddingRerankClient')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_config(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
        
class EmbeddingModel(BaseModel):
    def __init__(self, embedding_model, config_path='./configs/models.yaml'):
        super().__init__('embedding_models', embedding_model, config_path)
    
    async def embed_query(self, query=None):
        response = await self.client.embeddings.create(
            input=query,
            model=self.model
        )
        return response.data[0].embedding
    
    async def embed_documents(self, documents=None):
        response = await self.client.embeddings.create(
            input=documents,
            model=self.model
        )
        embs =  [response.data[i].embedding for i in range(len(response.data))]
        return embs

class RerankingModel(BaseModel):
    def __init__(self, reranking_model, config_path='./llm_tools/configs/models.yaml'):
        if "qwen" in reranking_model.lower():
            self.config = self.load_config(config_path)
            self.model_config = self.config['reranking_models'][reranking_model]
            self.model = self.model_config['model']
            self.logger = self._setup_logger()
            self.local_api_key = self.model_config['local_api_key']
            self.local_base_url = self.model_config['local_base_url'] + "/rerank"
            self.logger.info(f'[RerankingModel] Initializing Model: {reranking_model}')
            self.client_type = 'vllm'
            timeout = httpx.Timeout(30.0)
            self.http_client = httpx.AsyncClient(
                timeout=timeout,
                http2=True,
                headers={
                    "Content-Type": "application/json"
                }
            )
        else:
            super().__init__('reranking_models', reranking_model, config_path)
            self.client_type = 'openai'
    
    async def rerank_query(self, input=None, query=None):
        if self.client_type == 'openai':
            response = await self.client.embeddings.create(
                model=self.model,
                input=input,
                extra_body={"query": query}
            )
            return response.data[0].embedding
        elif self.client_type == 'vllm':
            payload = {
                "model": self.model,
                "encoding_format": "float",
                "query": query,
                "documents": [input]
                }

            try:
                async with asyncio.timeout(35.0):
                    response = await self.http_client.post(self.local_base_url, headers=self.http_client.headers, json=payload)
                    json_response = response.json()
                    results = json_response.get('results', [])[0]
                    return results.get('relevance_score')
            except asyncio.TimeoutError:
                self.logger.error("Request to reranking model timed out.")
                raise
            
    async def rerank_documents(self, documents=None, query=None):
        if self.client_type == 'openai':
            response = await self.client.embeddings.create(
                model=self.model,
                input=documents,
                extra_body={"query": query}
            )
            scores = [response.data[i].embedding for i in range(len(response.data))]
            return scores
        elif self.client_type == 'vllm':
            payload = {
                "model": self.model,
                "encoding_format": "float",
                "query": query,
                "documents": [doc for doc in documents]
                }
            
            try:
                async with asyncio.timeout(35.0):
                    response = await self.http_client.post(self.local_base_url, headers=self.http_client.headers, json=payload)
                    json_response = response.json()
                    results = json_response.get('results', [])
                    sorted_results = sorted(results, key=lambda item: item.get('index'))
                    scores = [result.get('relevance_score') for result in sorted_results]
                    return scores
            except asyncio.TimeoutError:
                self.logger.error("Request to reranking model timed out.")
                raise

class MultiVectorModel():
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.logger = self._setup_logger()
        self.logger.info(f'Initializing MultiVector Model from: {model_path}')
        from fastembed import LateInteractionTextEmbedding
        self.embedding_model = LateInteractionTextEmbedding(
            model_path, 
            cache_dir="./models",
            cuda=True,
            device_ids=[0],
        )
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def embed_query(self, query=None):
        embed = self.embedding_model.query_embed(query)
        return list(embed)[0]
    
    async def embed_query_batch(self, queries=None):
        embeds = self.embedding_model.query_embed(queries)
        embeds = list(embeds)
        return embeds
    
    async def embed_documents(self, documents=None):
        embeds = list(self.embedding_model.embed(documents))
        return embeds

class JinaForRerankingModel():
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.logger = self._setup_logger()
        self.logger.info(f'Initializing JinaForReranking Model from: {model_path}')
        import torch
        from .jina.jina_for_ranking import JinaForRanking
        self.model = JinaForRanking.from_pretrained(model_path, dtype="auto").to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()                         
    
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def rerank_documents(self, documents=None, query=None):
        respoonse = self.model.rerank_batch(
            [query],
            [documents],
            batch_size=1
        )
        result = respoonse[0]
        scores = [r['relevance_score'] for r in result]
        return scores   
    
    async def rerank_documents_batch(self, documents_list=None, query_list=None):
        response = self.model.rerank_batch(
            query_list,
            documents_list,
            batch_size=1
        )
        results = []
        for result in response:
            scores = [r['relevance_score'] for r in result]
            results.append(scores)
        return results