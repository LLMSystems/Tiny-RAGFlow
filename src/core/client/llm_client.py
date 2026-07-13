import base64
import imghdr
import os
import logging

import yaml
from openai import AsyncAzureOpenAI, AsyncOpenAI

from .llm_response_cache import LLMResponseCache


class AsyncLLMChat:
    def __init__(self, model, config_path='./configs/models.yaml', logger=None, cache_config=None):
        self.config = self.load_config(config_path)
        self.model_config = self.config['LLM_engines'][model]
        self.model = self.model_config['model']
        self.logger = logger if logger is not None else self._setup_logger()
        self.logger.info(f'[LLMChat] Initializing LLMChat with model: {model}')
        self.client = self._initialize_client()
        
        self.enable_cache = cache_config.get('enable', True) if cache_config else False
        if self.enable_cache:
            cache_file = cache_config.get('cache_file', './llm_cache.json')
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            self.cache = LLMResponseCache(cache_file=cache_file)
            self.logger.info(f'[LLMChat] LLM response caching enabled. Cache file: {cache_file}')

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
        
    def load_config(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
     
    def _initialize_client(self):
        self.translate_to_cht = self.model_config.get('translate_to_cht', False)
        if self.translate_to_cht:
                
            from opencc import OpenCC
            
            self.logger.info(f'[LLMChat] Translation to traditional Chinese enabled')
            self.zn_converter = OpenCC('s2twp')
        if 'gpt' in self.model and "oss" not in self.model and self.model_config.get('azure_api_key', None):
            self.logger.info(f'[LLMChat] Initializing AzureOpenAI client')
            return AsyncAzureOpenAI(
                azure_endpoint=self.model_config['azure_api_base'],
                api_key=self.model_config['azure_api_key'],
                api_version=self.model_config['azure_api_version']
            )
        else:
            self.logger.info(f'[LLMChat] Initializing Local OpenAI client')
            return AsyncOpenAI(
                api_key=self.model_config['local_api_key'],
                base_url=self.model_config['local_base_url']
            )        
    
    def initialize_history(self, system_message, user_message):
        if system_message is None:
            return [{"role": "user", "content": user_message}]
        else:
            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        
    async def chat(self, query, history=[], system='你是一個專業的助手，會用繁體中文回答問題。', params=None, response_format=None, stream=False, extra_body=None, multi_response=False, include_reasoning=False):
        if multi_response:
            self.logger.info(f'[LLMChat] Multi-response mode enabled, You can pass n (in params) parameter to specify the number of responses to return.')
            self.logger.info(f'[LLMChat] Stream mode disabled')
            stream = False
        
        if not history:
            self.logger.info(f'[LLMChat] Initializing history')
            history = self.initialize_history(system, query)
        else:
            history.append({'role': 'user', 'content': query})

        if params is None:
            self.logger.info(f'[LLMChat] Using default parameters')
            params = {
                'temperature': self.config['params']['default']['temperature'],
                'max_tokens': self.config['params']['default']['max_tokens'],
                'top_p': self.config['params']['default']['top_p'],
                'frequency_penalty': self.config['params']['default']['frequency_penalty'],
                'presence_penalty': self.config['params']['default']['presence_penalty']
            }
        else:
            params = params

        if extra_body is not None:
            params['extra_body'] = extra_body

        # detect n in params
        if params.get('n', 1) > 1 and not multi_response:
            self.logger.info(f'[LLMChat] please turn on multi-response mode to get multi responses.')

        # Check cache
        cache_key = self.cache.make_key(self.model, history, params) if self.enable_cache else None
        if self.enable_cache:
            cached = await self.cache.get(cache_key)
            if cached:
                self.logger.info(f'[LLMChat] Cache hit for key: {cache_key}')
                return cached['return'], history
        
        completion = await self.client.chat.completions.create(  
            model=self.model,
            stream=stream,
            response_format={'type': response_format} if response_format is not None else None,
            messages=history,
            **params
        )

        if stream:
            return self._handle_stream_response(completion, include_reasoning=include_reasoning)
        response = await self._handle_response(completion, multi_response=multi_response, include_reasoning=include_reasoning)
        
        if self.enable_cache:
            await self.cache.set(
                cache_key,
                result=response,
                model=self.model
            )
        
        return response, history

    async def _handle_response(self, completion, multi_response=False, include_reasoning=False):  
        if multi_response:
            responses = []
            for choice in completion.choices:
                response = choice.message.content
                if include_reasoning:
                    reasoning_content = getattr(choice.message, 'reasoning_content', '')
                    if reasoning_content:
                        response = f"\n<think>\n{reasoning_content}\n</think>\n" + response
                responses.append(self._maybe_translate(response))
            return responses
        else:
            response = completion.choices[0].message.content
            if include_reasoning:
                reasoning_content = getattr(completion.choices[0].message, 'reasoning_content', '')
                if reasoning_content:
                    response = f"\n<think>\n{reasoning_content}\n</think>\n" + response
            return self._maybe_translate(response)

    async def _handle_stream_response(self, completion, include_reasoning=False):  
        reasoning_started = False
        async for chunk in completion:
            try:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                content = delta.content
                if content:
                    if reasoning_started:
                        content = "\n</think>\n" + content
                        reasoning_started = False
                    translated_content = self._maybe_translate(content)
                    yield translated_content
                
                reasoning_content = getattr(delta, 'reasoning_content', None)
                if reasoning_content and include_reasoning:
                    translated_reasoning_content = self._maybe_translate(reasoning_content)
                    if not reasoning_started:
                        translated_reasoning_content = "\n<think>\n" + translated_reasoning_content
                        reasoning_started = True
                        yield translated_reasoning_content
                    else:
                        yield translated_reasoning_content
            
            except Exception as e:
                self.logger.error(f"Streaming error: {e}")

    def _maybe_translate(self, content):
        if self.translate_to_cht:
            return self.zn_converter.convert(content)
        return content

    def _image_to_base64(self, image_path):
        """Convert image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _detect_image_type(self, image_path):
        """Detect image type using imghdr."""
        with open(image_path, "rb") as image_file:
            return imghdr.what(image_file)

    def prepare_image_base64(self, image_path):
        """
        Prepare base64 encoded image with data URI prefix for reuse.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            str: Base64 encoded image with data URI prefix.
        """
        try:
            image_base64 = self._image_to_base64(image_path)
            image_type = self._detect_image_type(image_path)
            if image_type is None:
                raise ValueError("Unsupported image type or file is not an image.")
            return f"data:image/{image_type};base64,{image_base64}"
        except Exception as e:
            self.logger.error(f"Error preparing image base64: {e}")
            raise

    async def vision_chat(self, query, image_path=None, image_base64_with_prefix=None, history=[], system='你是一個專業的助手，會用繁體中文回答問題。', params=None, response_format=None, stream=False, extra_body=None):
        """
        Process an image and get a response from the vision LLM asynchronously.

        Args:
            query (str): The query to be sent to the chat completion API.
            image_path (str, optional): Path to the image file.
            image_base64_with_prefix (str, optional): Base64 encoded image with data URI prefix.
            history (list): Conversation history.
            system (str): System message.
            params (dict): Parameters for the API call.
            response_format (str): Response format type.
            stream (bool): Whether to stream the response.
            extra_body (dict): Extra body parameters.

        Returns:
            tuple: (response, history) if not streaming, otherwise async generator for streaming.
        
        Note:
            Either image_path or image_base64_with_prefix must be provided, but not both.
        """
        # Validate input parameters
        if image_path is None and image_base64_with_prefix is None:
            raise ValueError("Either image_path or image_base64_with_prefix must be provided.")
        if image_path is not None and image_base64_with_prefix is not None:
            raise ValueError("Cannot provide both image_path and image_base64_with_prefix. Choose one.")
        
        # Process image based on input type
        if image_path is not None:
            # Convert image file to base64
            try:
                image_base64 = self._image_to_base64(image_path)
                image_type = self._detect_image_type(image_path)
                if image_type is None:
                    raise ValueError("Unsupported image type or file is not an image.")
                image_base64_with_prefix = f"data:image/{image_type};base64,{image_base64}"
            except Exception as e:
                self.logger.error(f"Error processing image file: {e}")
                raise
        else:
            # Use provided base64 string directly
            if not image_base64_with_prefix.startswith('data:image/'):
                raise ValueError("image_base64_with_prefix must start with 'data:image/' prefix.")
            self.logger.info(f'[LLMChat] Using provided base64 image data')

        # Prepare message with image
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": image_base64_with_prefix}},
            ],
        }

        if not history:
            self.logger.info(f'[LLMChat] Initializing history for vision chat')
            history = [{'role': 'system', 'content': system}]
        
        history.append(user_message)

        completion = await self.client.chat.completions.create(
            model=self.model,
            stream=stream,
            response_format={'type':response_format} if response_format is not None else None,
            messages=history,
        )
        
        if stream:
            return self._handle_stream_response(completion)  
        else:
            response = await self._handle_response(completion)
            return response, history