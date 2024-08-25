#!/usr/bin/env python3

import logging
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote_plus
import asyncio
import re
from openai import AsyncOpenAI
import json
import os
import random
from aiocache import Cache, cached
from aiocache.serializers import PickleSerializer
from typing import List, Dict, Any
from config import OPENAI_API_KEY, ENGINE, MAX_TOKENS, TEMPERATURE, TOP_P, FREQUENCY_PENALTY, PRESENCE_PENALTY, TIMEOUT
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cache = Cache(Cache.MEMORY, serializer=PickleSerializer())

class WebScraper:
    def __init__(self):
        self.aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model_name = ENGINE
        self.learning_file = "scraper_learning.json"
        self.learned_data = self.load_learning()
        self.session = None
        self.connection_pool = None
        self.request_timeout = aiohttp.ClientTimeout(total=TIMEOUT)
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]

    def load_learning(self):
        if os.path.exists(self.learning_file):
            with open(self.learning_file, 'r') as f:
                logger.debug(f"Loading learned data from {self.learning_file}")
                return json.load(f)
        logger.debug(f"No learned data found, starting fresh.")
        return {}

    def save_learning(self):
        with open(self.learning_file, 'w') as f:
            json.dump(self.learned_data, f)
            logger.debug(f"Saved learning data to {self.learning_file}")

    async def create_session(self):
        if not self.connection_pool:
            self.connection_pool = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
        if not self.session:
            self.session = aiohttp.ClientSession(connector=self.connection_pool, timeout=self.request_timeout)

    async def close_session(self):
        if self.session:
            await self.session.close()
        if self.connection_pool:
            await self.connection_pool.close()
        self.session = None
        self.connection_pool = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def scrape(self, query, num_results=5):
        await self.create_session()
        try:
            logger.debug(f"Starting scrape for query: {query} with {num_results} results")
            google_results = await self.search_google(query, num_results)
            wiki_results = await self.search_wikipedia(query, 2)
            
            all_results = google_results + wiki_results
            logger.debug(f"Collected {len(all_results)} results from Google and Wikipedia")

            if not all_results:
                logger.warning("No results found from either Google or Wikipedia")
                return "I'm sorry, but I couldn't find any relevant information for your query."

            scraped_data = await asyncio.gather(*[
                self.process_result(query, result)
                for result in all_results
            ])

            valid_data = [data for data in scraped_data if data]
            logger.debug(f"Processed {len(valid_data)} valid results")
            
            if not valid_data:
                logger.warning("No valid data after processing results")
                return "I'm sorry, but I couldn't find any reliable information for your query."

            comprehensive_answer = await self.generate_comprehensive_answer(query, valid_data)
            return comprehensive_answer
        finally:
            await self.close_session()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_google(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        await asyncio.sleep(random.uniform(1, 3))
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results*2}"
        headers = {"User-Agent": random.choice(self.user_agents)}

        try:
            logger.debug(f"Searching Google with URL: {search_url}")
            async with self.session.get(search_url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Google search returned status code {response.status}")
                    return []
                content = await response.text()

            soup = BeautifulSoup(content, 'html.parser')
            search_results = []

            for g in soup.find_all('div', class_='g'):
                anchors = g.find_all('a')
                if anchors:
                    link = anchors[0]['href']
                    title = g.find('h3')
                    title = title.text if title else 'No title'
                    snippet = g.find('div', class_='VwiC3b')
                    snippet = snippet.text if snippet else 'No snippet'
                    
                    if link.startswith('/url?') or link.startswith('/search?'):
                        continue

                    if self.is_valid_url(link):
                        search_results.append({
                            'title': title,
                            'link': link,
                            'snippet': snippet
                        })

                    if len(search_results) >= num_results:
                        break

            logger.debug(f"Found {len(search_results)} results from Google search")
            return search_results

        except Exception as e:
            logger.error(f"Unexpected error during Google search: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_wikipedia(self, query, num_results=2):
        await asyncio.sleep(random.uniform(1, 3))
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote_plus(query)}&limit={num_results}&namespace=0&format=json"
        
        try:
            logger.debug(f"Searching Wikipedia with URL: {search_url}")
            async with self.session.get(search_url) as response:
                if response.status != 200:
                    logger.error(f"Wikipedia search returned status code {response.status}")
                    return []
                data = await response.json()
            
            titles, _, urls = data[1], data[2], data[3]
            results = []
            
            for title, url in zip(titles, urls):
                snippet = await self.get_wikipedia_snippet(url)
                if snippet:
                    results.append({
                        'title': title,
                        'link': url,
                        'snippet': snippet
                    })
            
            logger.debug(f"Collected {len(results)} results from Wikipedia")
            return results

        except Exception as e:
            logger.error(f"Error during Wikipedia search: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_wikipedia_snippet(self, url):
        try:
            logger.debug(f"Fetching Wikipedia snippet from URL: {url}")
            async with self.session.get(url) as response:
                content = await response.text()
                logger.debug(f"Wikipedia snippet response status: {response.status}")
            
            soup = BeautifulSoup(content, 'html.parser')
            paragraphs = soup.find_all('p')
            
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 50:
                    snippet = text[:500] + "..."
                    logger.debug(f"Extracted snippet: {snippet}")
                    return snippet
            
            logger.debug("No suitable snippet found.")
            return "No snippet available"

        except Exception as e:
            logger.error(f"Error fetching Wikipedia snippet: {str(e)}")
            return "Error fetching snippet"

    def is_valid_url(self, url):
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @cached(ttl=3600)
    async def fetch_page_content(self, url):
        try:
            logger.debug(f"Fetching content from URL: {url}")
            async with self.session.get(url, timeout=20) as response:
                content = await response.text()
            soup = BeautifulSoup(content, 'html.parser')
            page_text = ' '.join([p.get_text() for p in soup.find_all('p')])
            logger.debug(f"Fetched content length: {len(page_text)} characters")
            return page_text
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return ""

    def clean_content(self, content):
        cleaned_content = re.sub(r'\s+', ' ', re.sub(r'\[\d+\]', '', content)).strip()
        logger.debug(f"Cleaned content: {cleaned_content[:100]}...")
        return cleaned_content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def check_relevance(self, query, content):
        prompt = f"Query: {query}\nContent: {content[:1000]}\nIs the content relevant to the query? Answer with a number between 0 and 1."
        relevance_score = await self.get_openai_response(prompt, default_value=0.5)
        logger.debug(f"Relevance score for query '{query}': {relevance_score}")
        return relevance_score

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def run_fact_checker(self, query, content):
        prompt = f"Query: {query}\nContent: {content[:1000]}\nDoes this content contain factual information related to the query? Answer with 'Yes' or 'No'."
        response = await self.get_openai_response(prompt, default_value='no')
        is_factual = response.lower() == 'yes'
        logger.debug(f"Fact-check result for query '{query}': {is_factual}")
        return is_factual

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_openai_response(self, prompt, default_value):
        try:
            response = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                frequency_penalty=FREQUENCY_PENALTY,
                presence_penalty=PRESENCE_PENALTY
            )
            response_content = response.choices[0].message.content.strip()
            logger.debug(f"OpenAI response: {response_content}")
            return response_content
        except Exception as e:
            logger.error(f"Error getting OpenAI response: {str(e)}")
            return default_value

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def process_result(self, query, result):
        try:
            logger.debug(f"Processing result from {result['link']}")
            page_content = await self.fetch_page_content(result['link'])
            if page_content:
                page_content = self.clean_content(page_content)
                relevance_score = await self.check_relevance(query, page_content)
                relevance_score = float(relevance_score)
                is_factual = await self.run_fact_checker(query, page_content)
                if relevance_score > 0.5 and is_factual:
                    await self.learn(query, page_content)
                    logger.debug(f"Valid result processed from {result['link']}")
                    return {
                        'title': result['title'],
                        'content': page_content,
                        'url': result['link']
                    }
            logger.debug(f"Result from {result['link']} did not meet relevance or factual criteria")
            return None
        except Exception as e:
            logger.error(f"Error processing result from {result['link']}: {str(e)}")
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_comprehensive_answer(self, query, valid_data):
        try:
            combined_content = "\n\n".join(
                data['content'] for data in valid_data if isinstance(data.get('content'), str)
            )
            
            prompt = (
                f"Based on the following information, provide a comprehensive and accurate answer to the question: "
                f"'{query}'\n\nInformation:\n{combined_content}\n\nPlease synthesize the information and provide a single, coherent answer."
            )

            response = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate and concise information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                frequency_penalty=FREQUENCY_PENALTY,
                presence_penalty=PRESENCE_PENALTY
            )
            
            answer = response.choices[0].message.content.strip()
            logger.debug(f"Generated comprehensive answer: {answer[:100]}...")
            return answer
        except Exception as e:
            logger.error(f"Error generating comprehensive answer: {str(e)}")
            return "I'm sorry, but I couldn't generate a comprehensive answer based on the available information."

    async def learn(self, query, content):
        prompt = f"Extract 3 key points from this content related to the query '{query}': {content[:1000]}"
        key_points = await self.get_openai_response(prompt, default_value='')
        key_points = key_points.split('\n')
        self.learned_data.setdefault(query, []).extend(key_points[-10:])
        self.save_learning()
        logger.debug(f"Learned key points for query '{query}': {key_points}")

    def get_learned_info(self, query):
        learned_info = self.learned_data.get(query, [])
        logger.debug(f"Retrieved learned info for query '{query}': {learned_info}")
        return learned_info

async def search_internet_and_process(query):
    scraper = WebScraper()
    await scraper.create_session()
    try:
        comprehensive_answer = await scraper.scrape(query)
        logger.debug(f"Final comprehensive answer: {comprehensive_answer[:100]}...")
        return {
            'answer': comprehensive_answer,
            'success': comprehensive_answer != "I'm sorry, but I couldn't find any relevant information for your query."
        }
    finally:
        await scraper.close_session()

if __name__ == "__main__":
    async def main():
        query = "Learn to program Bash scripts"
        result = await search_internet_and_process(query)
        if result['success']:
            print(f"Answer: {result['answer']}")
        else:
            print("No relevant information found.")

    asyncio.run(main())