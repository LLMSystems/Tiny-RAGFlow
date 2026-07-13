import asyncio
import json
from .prompt import faq_generation_prompt
from ..core.client.llm_client import AsyncLLMChat
from ..utils.utils import _normalize_json_response


async def generate_faq_stage1(llm_client: AsyncLLMChat, answer_text: str, params=None) -> str:
    if params is None:
        params = {
            "temperature": 0.1
        }
    prompt = faq_generation_prompt.format(ANSWER_TEXT=answer_text)
    response, _ = await llm_client.chat(
        query=prompt,
        params=params
    )
    
    response_clean = _normalize_json_response(response)
    try:
        response_json = json.loads(response_clean)
    except json.JSONDecodeError:
        import json_repair
        response_json = json_repair.loads(response_clean)
    
    return response_json