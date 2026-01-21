import asyncio
import json

from src.core.client.llm_client import AsyncLLMChat
from src.qa_generation.prompt import faq_generation_prompt

from src.qa_generation.stage_1 import generate_faq_stage1


llmchater = AsyncLLMChat(
            model="gpt-5-nano",
            config_path="./config/models.yaml",
            cache_config={
                "enable": True,
                "cache_file": './cache/llm_chat_cache.json'
            }
            )

data_path = "./data/bonnieQA/dataset.json"

with open(data_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)
    
async def main():
    data = dataset[10]
    answer_text = data["metadata"]['answer']
    print(answer_text)
    
    result = await generate_faq_stage1(
        llmchater,
        answer_text,
        params={
            "temperature": 1,
        }
    )
    print("\n==== Generated FAQ ====")
    print(result)
    
    
if __name__ == "__main__":
    asyncio.run(main())