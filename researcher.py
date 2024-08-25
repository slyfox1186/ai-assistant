#!/usr/bin/env python3

import aiohttp
import asyncio
import logging
from typing import List, Dict
from web_scraper import WebScraper, search_internet_and_process
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI
import time
from config import OPENAI_API_KEY, ENGINE, MAX_TOKENS, TEMPERATURE, TOP_P, FREQUENCY_PENALTY, PRESENCE_PENALTY, TIMEOUT
from data_manager import DataManager

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Researcher:
    def __init__(self):
        self.web_scraper = WebScraper()
        self.session = None
        self.rate_limit = asyncio.Semaphore(20)
        self.aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.data_manager = DataManager()
        self.total_tokens_used = 0
        self.last_execution_time = 0

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    def get_tokens_per_iteration(self, iterations: int) -> int:
        remaining_tokens = MAX_TOKENS - self.total_tokens_used
        return max(remaining_tokens // iterations, 1)

    async def enforce_rate_limit(self):
        async with self.rate_limit:
            time_since_last_execution = time.time() - self.last_execution_time
            if time_since_last_execution < 1:
                await asyncio.sleep(1 - time_since_last_execution)
            self.last_execution_time = time.time()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def research_topic(self, query: str, iterations: int) -> List[Dict[str, str]]:
        all_results = []
        current_query = query

        for i in range(iterations):
            await self.enforce_rate_limit()
            logger.info(f"Starting iteration {i+1}/{iterations} for query: {current_query}")
            
            try:
                scraped_data = await self.scrape_with_retries(current_query)
                if isinstance(scraped_data, dict) and 'answer' in scraped_data and scraped_data['answer']:
                    result = {
                        'title': f"Research Result {i+1}",
                        'summary': scraped_data['answer']
                    }
                else:
                    result = {
                        'title': f"Research Result {i+1}",
                        'summary': f"I'm sorry, but I couldn't find any relevant information for the query: {current_query}"
                    }
                all_results.append(result)
                
                # Refine the query for the next iteration
                current_query = await self.refine_query(query, current_query, all_results, self.get_tokens_per_iteration(iterations - i))
            
            except Exception as e:
                logger.error(f"Error during research iteration: {str(e)}")
                result = {
                    'title': f"Research Result {i+1}",
                    'summary': f"An error occurred while researching the query: {current_query}. Error: {str(e)}"
                }
                all_results.append(result)

            logger.info(f"Completed iteration {i+1}/{iterations}")

        return all_results

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def scrape_with_retries(self, query: str) -> Dict[str, str]:
        try:
            scraped_data = await search_internet_and_process(query)
            if not scraped_data or not isinstance(scraped_data, dict) or 'answer' not in scraped_data:
                return {"answer": f"No relevant information found for query: {query}"}
            return scraped_data
        except Exception as e:
            logger.error(f"Error in scrape_with_retries for query '{query}': {str(e)}")
            return {"answer": f"Error occurred while searching for information: {str(e)}"}

    async def refine_query(self, original_query: str, current_query: str, batch_results: List[Dict[str, str]], tokens_per_iteration: int) -> str:
        summaries = "\n".join([result['summary'] for result in batch_results])
        try:
            response = await self.aclient.chat.completions.create(
                model=ENGINE,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that refines search queries based on previous results and the original query intent."},
                    {"role": "user", "content": f"Original query: {original_query}\nCurrent query: {current_query}\nPrevious results:\n{summaries}\nProvide a refined search query to explore this topic further and uncover new information."}
                ],
                max_tokens=tokens_per_iteration,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                frequency_penalty=FREQUENCY_PENALTY,
                presence_penalty=PRESENCE_PENALTY,
                timeout=TIMEOUT
            )
            refined_query = response.choices[0].message.content.strip()
            self.total_tokens_used += response.usage.total_tokens
            return refined_query
        except Exception as e:
            logger.error(f"Error refining query: {str(e)}")
            return current_query

    async def expand_query(self, original_query: str, current_query: str, tokens_per_iteration: int) -> str:
        try:
            response = await self.aclient.chat.completions.create(
                model=ENGINE,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that expands search queries to find more information while maintaining the original query intent."},
                    {"role": "user", "content": f"Original query: {original_query}\nCurrent query: {current_query}\nThe current query didn't yield new results. Provide an expanded search query that might yield better results."}
                ],
                max_tokens=tokens_per_iteration,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                frequency_penalty=FREQUENCY_PENALTY,
                presence_penalty=PRESENCE_PENALTY,
                timeout=TIMEOUT
            )
            expanded_query = response.choices[0].message.content.strip()
            self.total_tokens_used += response.usage.total_tokens
            return expanded_query
        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}")
            return f"Advanced techniques and practical examples for {original_query}"

    async def generate_final_summary(self, original_query: str, results: List[Dict[str, str]]) -> str:
        try:
            valid_results = [
                result for result in results
                if isinstance(result, dict) and 'summary' in result and 
                not result['summary'].startswith("I'm sorry") and
                not result['summary'].startswith("An error occurred")
            ]

            if not valid_results:
                return "No valid research results were found to generate a final summary. Please try refining your query or check your internet connection."

            all_summaries = "\n\n".join([f"Result {i+1}: {result['summary']}" for i, result in enumerate(valid_results)])

            prompt = (
                f"Original query: {original_query}\n\n"
                f"Research results:\n{all_summaries}\n\n"
                "Please provide a comprehensive summary that synthesizes the key information from the research results above to address the original query. "
                "If no relevant information was found, suggest ways to refine the query or alternative approaches to find the information."
            )

            response = await self.aclient.chat.completions.create(
                model=ENGINE,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that summarizes research results."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=min(3000, MAX_TOKENS - self.total_tokens_used),
                temperature=TEMPERATURE,
                top_p=TOP_P,
                frequency_penalty=FREQUENCY_PENALTY,
                presence_penalty=PRESENCE_PENALTY,
                timeout=TIMEOUT
            )
            final_summary = response.choices[0].message.content.strip()
            self.total_tokens_used += response.usage.total_tokens
            return final_summary

        except Exception as e:
            logger.error(f"Error generating final summary: {str(e)}")
            return "An error occurred while generating the final summary. Please try your query again or refine it for better results."

    def format_research_results(self, results: List[Dict[str, str]], final_summary: str) -> str:
        if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
            logger.error(f"Expected results to be a list of dictionaries, but got {type(results).__name__}")
            return "An error occurred while processing research results."

        formatted_results = "[Research Results]\n\n"
        for i, result in enumerate(results):
            if isinstance(result, dict) and 'summary' in result:
                formatted_results += f"Title: {result.get('title', 'Unknown')}\n"
                formatted_results += f"Summary: {result['summary']}\n"
                formatted_results += "-" * 40 + "\n\n"
            else:
                logger.warning(f"Invalid result format at index {i}: {result}")
                continue  # Skip invalid entries

        formatted_results += f"Final Summary:\n{final_summary}\n"
        return formatted_results

async def research_topic(query: str, iterations: int) -> List[Dict[str, str]]:
    researcher = Researcher()
    return await researcher.research_topic(query, iterations)