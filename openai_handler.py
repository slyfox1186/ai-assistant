#!/usr/bin/env python3

import openai
import asyncio
import aiohttp
import logging
import os
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import OPENAI_API_KEY, MAX_RETRIES, MAX_TOKENS, TEMPERATURE, TOP_P, FREQUENCY_PENALTY, PRESENCE_PENALTY, TIMEOUT, CACHE_SIZE, MAX_WORKERS
from aiocache import cached
from aiocache.serializers import PickleSerializer
import time

# Set up logging with a clear format for easier debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = None

def set_openai_api_key(api_key: str):
    global client
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        client = openai.AsyncOpenAI(api_key=api_key)
        logger.info(f"OpenAI API key set successfully.")
    else:
        logger.error("OpenAI API key is not set.")
        raise ValueError("OpenAI API key must be set.")

class OpenAIHandler:
    def __init__(self):
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.semaphore = asyncio.Semaphore(MAX_WORKERS)
        
        # Ensure the API key is set at initialization
        if not client:
            set_openai_api_key(OPENAI_API_KEY)
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=TIMEOUT)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.RateLimitError, openai.APIConnectionError)),
        after=lambda retry_state: logger.error(f"Attempt {retry_state.attempt_number} failed: {retry_state.outcome.exception()}")
    )
    async def get_openai_response(self, query: str, **kwargs) -> str:
        if client is None:
            raise ValueError("OpenAI API key not set.")
        
        async with self.semaphore:
            logger.info(f"Sending query to OpenAI: {query}")
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ],
                max_tokens=kwargs.get('max_tokens', MAX_TOKENS),
                temperature=kwargs.get('temperature', TEMPERATURE),
                top_p=kwargs.get('top_p', TOP_P),
                frequency_penalty=kwargs.get('frequency_penalty', FREQUENCY_PENALTY),
                presence_penalty=kwargs.get('presence_penalty', PRESENCE_PENALTY)
            )
            logger.info("Received response from OpenAI.")
            return response.choices[0].message.content.strip()

    @cached(ttl=3600, key_builder=lambda *args, **kwargs: f"openai_response:{args[1]}", serializer=PickleSerializer())
    async def get_openai_response_cached(self, query: str, **kwargs) -> str:
        logger.info(f"Retrieving cached response for query: {query}")
        return await self.get_openai_response(query, **kwargs)

    async def batch_process_queries(self, queries: List[str], **kwargs) -> List[str]:
        logger.info("Processing batch of queries.")
        return await asyncio.gather(*[self.get_openai_response(query, **kwargs) for query in queries])

    async def stream_openai_response(self, query: str, **kwargs) -> AsyncGenerator[str, None]:
        if client is None:
            raise ValueError("OpenAI API key not set.")
        
        logger.info(f"Streaming response for query: {query}")
        try:
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ],
                max_tokens=kwargs.get('max_tokens', MAX_TOKENS),
                temperature=kwargs.get('temperature', TEMPERATURE),
                top_p=kwargs.get('top_p', TOP_P),
                frequency_penalty=kwargs.get('frequency_penalty', FREQUENCY_PENALTY),
                presence_penalty=kwargs.get('presence_penalty', PRESENCE_PENALTY),
                stream=True
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    logger.info(f"Streaming chunk received.")
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI API error during streaming: {e}")
            raise

    async def get_embedding(self, text: str) -> List[float]:
        if client is None:
            raise ValueError("OpenAI API key not set.")
        
        logger.info(f"Getting embedding for text.")
        try:
            response = await client.embeddings.create(input=text, model="text-embedding-ada-002")
            logger.info("Received embedding from OpenAI.")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI API error while getting embedding: {e}")
            raise

    async def batch_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info("Batch processing embeddings.")
        return await asyncio.gather(*[self.get_embedding(text) for text in texts])

    async def moderate_content(self, text: str) -> Dict[str, Any]:
        if client is None:
            raise ValueError("OpenAI API key not set.")
        
        logger.info("Moderating content.")
        try:
            response = await client.moderations.create(input=text)
            logger.info("Content moderation completed.")
            return response.results[0].model_dump()
        except Exception as e:
            logger.error(f"OpenAI API error during content moderation: {e}")
            raise

    async def batch_moderate_content(self, texts: List[str]) -> List[Dict[str, Any]]:
        logger.info("Batch moderating content.")
        return await asyncio.gather(*[self.moderate_content(text) for text in texts])

# Instantiate the handler
openai_handler = OpenAIHandler()

async def get_openai_response(query: str, **kwargs) -> str:
    return await openai_handler.get_openai_response_cached(query, **kwargs)

async def batch_process_queries(queries: List[str], **kwargs) -> List[str]:
    return await openai_handler.batch_process_queries(queries, **kwargs)

async def stream_openai_response(query: str, **kwargs) -> AsyncGenerator[str, None]:
    return openai_handler.stream_openai_response(query, **kwargs)

async def get_embedding(text: str) -> List[float]:
    return await openai_handler.get_embedding(text)

async def batch_get_embeddings(texts: List[str]) -> List[List[float]]:
    return await openai_handler.batch_get_embeddings(texts)

async def moderate_content(text: str) -> Dict[str, Any]:
    return await openai_handler.moderate_content(text)

async def batch_moderate_content(texts: List[str]) -> List[Dict[str, Any]]:
    return await openai_handler.batch_moderate_content(texts)

class ResponseTimeTracker:
    def __init__(self):
        self.response_times = []

    def add_response_time(self, response_time: float):
        logger.info(f"Adding response time: {response_time:.2f} seconds")
        self.response_times.append(response_time)

    def get_average_response_time(self) -> float:
        if not self.response_times:
            logger.info("No response times recorded, returning 0.")
            return 0.0
        average_time = sum(self.response_times) / len(self.response_times)
        logger.info(f"Calculated average response time: {average_time:.2f} seconds")
        return average_time

    def get_max_response_time(self) -> float:
        if not self.response_times:
            logger.info("No response times recorded, returning 0.")
            return 0.0
        max_time = max(self.response_times)
        logger.info(f"Maximum response time: {max_time:.2f} seconds")
        return max_time

    def get_min_response_time(self) -> float:
        if not self.response_times:
            logger.info("No response times recorded, returning 0.")
            return 0.0
        min_time = min(self.response_times)
        logger.info(f"Minimum response time: {min_time:.2f} seconds")
        return min_time

response_tracker = ResponseTimeTracker()

async def get_openai_response_with_timing(query: str, **kwargs) -> Tuple[str, float]:
    logger.info("Measuring response time for query.")
    start_time = time.time()
    response = await openai_handler.get_openai_response(query, **kwargs)
    response_time = time.time() - start_time
    logger.info(f"OpenAI response time: {response_time:.2f} seconds")
    response_tracker.add_response_time(response_time)
    return response, response_time

def get_response_time_stats() -> Dict[str, float]:
    logger.info("Fetching response time statistics.")
    return {
        "average": response_tracker.get_average_response_time(),
        "max": response_tracker.get_max_response_time(),
        "min": response_tracker.get_min_response_time()
    }
