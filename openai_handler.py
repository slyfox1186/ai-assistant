#!/usr/bin/env python3

import openai
import asyncio
import aiohttp
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Generator
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import OPENAI_API_KEY, MAX_RETRIES, MAX_TOKENS, TEMPERATURE, TOP_P, FREQUENCY_PENALTY, PRESENCE_PENALTY, TIMEOUT, CACHE_SIZE, MAX_WORKERS
import ujson
from aiocache import cached
from aiocache.serializers import PickleSerializer

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

openai.api_key = OPENAI_API_KEY

def set_openai_api_key(api_key: str):
    openai.api_key = api_key

class OpenAIHandler:
    def __init__(self):
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.semaphore = asyncio.Semaphore(MAX_WORKERS)

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=TIMEOUT)
        self.session = aiohttp.ClientSession(json_serialize=ujson.dumps, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    @retry(stop=stop_after_attempt(MAX_RETRIES),
           wait=wait_exponential(multiplier=1, min=4, max=10),
           retry=retry_if_exception_type((openai.error.APIError, openai.error.RateLimitError, openai.error.ServiceUnavailableError)),
           after=lambda retry_state: logger.error(f"Attempt {retry_state.attempt_number} failed: {retry_state.outcome.exception()}"))
    async def get_openai_response(self, query: str, **kwargs) -> str:
        async with self.semaphore:
            try:
                logger.debug(f"Sending query to OpenAI: {query}")
                response = await openai.ChatCompletion.acreate(
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
                logger.debug(f"Received response from OpenAI: {response}")
                return response['choices'][0]['message']['content'].strip()
            except openai.error.OpenAIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise

    @cached(ttl=3600, key_builder=lambda *args, **kwargs: args[1], serializer=PickleSerializer())
    async def get_openai_response_cached(self, query: str, **kwargs) -> str:
        logger.debug(f"Retrieving cached response for query: {query}")
        return await self.get_openai_response(query, **kwargs)

    async def batch_process_queries(self, queries: List[str], **kwargs) -> List[str]:
        async def process_query(query):
            return await self.get_openai_response(query, **kwargs)

        return await asyncio.gather(*[process_query(query) for query in queries])

    async def stream_openai_response(self, query: str, **kwargs) -> Generator[str, None, None]:
        try:
            logger.debug(f"Streaming response for query: {query}")
            async for chunk in await openai.ChatCompletion.acreate(
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
            ):
                if chunk['choices'][0]['delta'].get('content'):
                    logger.debug(f"Streaming chunk: {chunk['choices'][0]['delta']['content']}")
                    yield chunk['choices'][0]['delta']['content']
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error during streaming: {e}")
            raise

    async def get_embedding(self, text: str) -> List[float]:
        try:
            logger.debug(f"Getting embedding for text: {text}")
            response = await openai.Embedding.acreate(
                input=text,
                model="text-embedding-ada-002"
            )
            logger.debug(f"Received embedding from OpenAI: {response['data'][0]['embedding']}")
            return response['data'][0]['embedding']
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error while getting embedding: {e}")
            raise

    async def batch_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.gather(*[self.get_embedding(text) for text in texts])

    async def moderate_content(self, text: str) -> Dict[str, Any]:
        try:
            logger.debug(f"Moderating content: {text}")
            response = await openai.Moderation.acreate(input=text)
            logger.debug(f"Moderation result: {response['results'][0]}")
            return response['results'][0]
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error during content moderation: {e}")
            raise

    async def batch_moderate_content(self, texts: List[str]) -> List[Dict[str, Any]]:
        return await asyncio.gather(*[self.moderate_content(text) for text in texts])

openai_handler = OpenAIHandler()

async def get_openai_response(query: str, **kwargs) -> str:
    return await openai_handler.get_openai_response_cached(query, **kwargs)

async def batch_process_queries(queries: List[str], **kwargs) -> List[str]:
    return await openai_handler.batch_process_queries(queries, **kwargs)

async def stream_openai_response(query: str, **kwargs) -> Generator[str, None, None]:
    return await openai_handler.stream_openai_response(query, **kwargs)

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
        logger.debug(f"Adding response time: {response_time} seconds")
        self.response_times.append(response_time)

    def get_average_response_time(self) -> float:
        if not self.response_times:
            logger.debug("No response times recorded, returning 0")
            return 0
        average_time = sum(self.response_times) / len(self.response_times)
        logger.debug(f"Calculated average response time: {average_time} seconds")
        return average_time

    def get_max_response_time(self) -> float:
        if not self.response_times:
            logger.debug("No response times recorded, returning 0")
            return 0
        max_time = max(self.response_times)
        logger.debug(f"Maximum response time: {max_time} seconds")
        return max_time

    def get_min_response_time(self) -> float:
        if not self.response_times:
            logger.debug("No response times recorded, returning 0")
            return 0
        min_time = min(self.response_times)
        logger.debug(f"Minimum response time: {min_time} seconds")
        return min_time

response_tracker = ResponseTimeTracker()

async def get_openai_response_with_timing(query: str, **kwargs) -> Tuple[str, float]:
    logger.debug(f"Getting OpenAI response with timing for query: {query}")
    start_time = time.time()
    response = await openai_handler.get_openai_response(query, **kwargs)
    end_time = time.time()
    response_time = end_time - start_time
    logger.debug(f"OpenAI response time: {response_time} seconds")
    response_tracker.add_response_time(response_time)
    return response, response_time

def get_response_time_stats() -> Dict[str, float]:
    logger.debug("Fetching response time statistics")
    return {
        "average": response_tracker.get_average_response_time(),
        "max": response_tracker.get_max_response_time(),
        "min": response_tracker.get_min_response_time()
    }

if __name__ == "__main__":
    async def main():
        async with OpenAIHandler() as handler:
            query = "What is the capital of France?"
            response = await handler.get_openai_response(query)
            print(f"Response: {response}")

            queries = ["What is Python?", "Who invented the telephone?"]
            responses = await handler.batch_process_queries(queries)
            for q, r in zip(queries, responses):
                print(f"Query: {q}\nResponse: {r}\n")

            moderation = await handler.moderate_content("This is a test.")
            print(f"Moderation result: {moderation}")

    asyncio.run(main())