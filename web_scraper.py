import logging
import aiohttp
from lxml import html
from urllib.parse import urlparse, quote_plus
import asyncio
import re
import openai
import json
import os
from aiocache import Cache, cached
from aiocache.serializers import PickleSerializer
from typing import List, Dict, Any

# Set logging level to DEBUG for maximum verbose output
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize cache
cache = Cache(Cache.MEMORY, serializer=PickleSerializer())

class WebScraper:
    def __init__(self, api_key=None):
        if api_key:
            openai.api_key = api_key
        self.model_name = "gpt-4o-mini"
        self.learning_file = "scraper_learning.json"
        self.learned_data = self.load_learning()
        self.session = None
        self.connection_pool = None
        self.request_timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout

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

    async def scrape(self, query, num_results=5):
        await self.create_session()
        logger.debug(f"Starting scrape for query: {query} with {num_results} results")
        google_results = await self.search_google(query, num_results)
        wiki_results = await self.search_wikipedia(query, 2)  # Limit Wikipedia results to 2
        
        all_results = google_results + wiki_results
        logger.debug(f"Collected {len(all_results)} results from Google and Wikipedia")

        scraped_data = await asyncio.gather(*[
            self.process_result(query, result)
            for result in all_results
        ])

        await self.close_session()
        valid_data = [data for data in scraped_data if data]
        logger.debug(f"Processed {len(valid_data)} valid results")
        
        comprehensive_answer = await self.generate_comprehensive_answer(query, valid_data)
        return comprehensive_answer

    async def search_google(self, query, num_results=5):
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            logger.debug(f"Searching Google with URL: {search_url}")
            async with self.session.get(search_url, headers=headers) as response:
                content = await response.text()
                logger.debug(f"Google search response status: {response.status}")
            tree = html.fromstring(content)

            results = []
            for g in tree.xpath('//div[@class="g"]'):
                title = g.xpath('.//h3/text()')[0] if g.xpath('.//h3') else "No title"
                link = self.clean_link(g.xpath('.//a/@href')[0]) if g.xpath('.//a/@href') else ""
                snippet = g.xpath('.//div[@class="VwiC3b"]/text()')[0] if g.xpath('.//div[@class="VwiC3b"]') else "No snippet"
                
                if self.is_valid_url(link):
                    results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })
                
                if len(results) >= num_results:
                    break

            logger.debug(f"Found {len(results)} results from Google search")
            return results

        except Exception as e:
            logger.error(f"Error during Google search: {str(e)}")
            return []

    async def search_wikipedia(self, query, num_results=2):
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote_plus(query)}&limit={num_results}&namespace=0&format=json"
        
        try:
            logger.debug(f"Searching Wikipedia with URL: {search_url}")
            async with self.session.get(search_url) as response:
                data = await response.json()
                logger.debug(f"Wikipedia search response status: {response.status}")
            
            titles, _, urls = data[1], data[2], data[3]
            results = []
            
            for title, url in zip(titles, urls):
                snippet = await self.get_wikipedia_snippet(url)
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

    async def get_wikipedia_snippet(self, url):
        try:
            logger.debug(f"Fetching Wikipedia snippet from URL: {url}")
            async with self.session.get(url) as response:
                content = await response.text()
                logger.debug(f"Wikipedia snippet response status: {response.status}")
            
            tree = html.fromstring(content)
            paragraphs = tree.xpath('//p')
            
            for p in paragraphs:
                text = p.text_content().strip()
                if len(text) > 50:
                    snippet = text[:500] + "..."
                    logger.debug(f"Extracted snippet: {snippet}")
                    return snippet
            
            logger.debug("No suitable snippet found.")
            return "No snippet available"

        except Exception as e:
            logger.error(f"Error fetching Wikipedia snippet: {str(e)}")
            return "Error fetching snippet"

    def clean_link(self, link):
        if link.startswith('/url?q='):
            clean_link = link.split('/url?q=')[1].split('&')[0]
            logger.debug(f"Cleaned link: {clean_link}")
            return clean_link
        logger.debug(f"Returning original link: {link}")
        return link

    def is_valid_url(self, url):
        parsed = urlparse(url)
        valid = bool(parsed.netloc) and bool(parsed.scheme)
        logger.debug(f"URL validation for {url}: {valid}")
        return valid

    @cached(ttl=3600)
    async def fetch_page_content(self, url):
        try:
            logger.debug(f"Fetching content from URL: {url}")
            async with self.session.get(url, timeout=10) as response:
                content = await response.text()
            tree = html.fromstring(content)
            page_text = ' '.join(tree.xpath('//p/text()'))
            logger.debug(f"Fetched content length: {len(page_text)} characters")
            return page_text
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return ""

    def clean_content(self, content):
        cleaned_content = re.sub(r'\s+', ' ', re.sub(r'\[\d+\]', '', content)).strip()
        logger.debug(f"Cleaned content: {cleaned_content[:100]}...")  # Show only the first 100 characters
        return cleaned_content

    async def check_relevance(self, query, content):
        prompt = f"Query: {query}\nContent: {content[:1000]}\nIs the content relevant to the query? Answer with a number between 0 and 1."
        relevance_score = await self.get_openai_response(prompt, default_value=0.5)
        logger.debug(f"Relevance score for query '{query}': {relevance_score}")
        return relevance_score

    async def run_fact_checker(self, query, content):
        prompt = f"Query: {query}\nContent: {content[:1000]}\nDoes this content contain factual information related to the query? Answer with 'Yes' or 'No'."
        response = await self.get_openai_response(prompt, default_value='no')
        is_factual = response.lower() == 'yes'
        logger.debug(f"Fact-check result for query '{query}': {is_factual}")
        return is_factual

    async def get_openai_response(self, prompt, default_value):
        if not openai.api_key:
            logger.warning("OpenAI API key not set. Skipping OpenAI processing.")
            return default_value
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1,
                n=1,
                stop=None,
                temperature=0.5
            )
            response_content = response.choices[0].message['content'].strip()
            logger.debug(f"OpenAI response: {response_content}")
            return response_content
        except Exception as e:
            logger.error(f"Error getting OpenAI response: {str(e)}")
            return default_value

    async def process_result(self, query, result):
        try:
            logger.debug(f"Processing result from {result['link']}")
            page_content = await self.fetch_page_content(result['link'])
            if page_content:
                page_content = self.clean_content(page_content)
                if openai.api_key:
                    relevance_score = await self.check_relevance(query, page_content)
                    relevance_score = float(relevance_score)  # Convert relevance_score to float
                    is_factual = await self.run_fact_checker(query, page_content)
                    if relevance_score > 0.5 and is_factual:
                        await self.learn(query, page_content)
                        logger.debug(f"Valid result processed from {result['link']}")
                        return {
                            'title': result['title'],
                            'content': page_content,
                            'url': result['link']
                        }
                else:
                    logger.debug(f"Returning result without AI processing: {result['link']}")
                    return {
                        'title': result['title'],
                        'content': page_content,
                        'url': result['link']
                    }
        except Exception as e:
            logger.error(f"Error processing result from {result['link']}: {str(e)}")
        return None

    async def generate_comprehensive_answer(self, query, valid_data):
        combined_content = "\n\n".join([data['content'] for data in valid_data])
        prompt = f"Based on the following information, provide a comprehensive and accurate answer to the question: '{query}'\n\nInformation:\n{combined_content}\n\nPlease synthesize the information and provide a single, coherent answer."
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate and concise information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.7
            )
            answer = response.choices[0].message['content'].strip()
            logger.debug(f"Generated comprehensive answer: {answer[:100]}...")  # Log first 100 characters
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
    scraper = WebScraper(api_key=os.getenv('OPENAI_API_KEY'))  # Initialize the scraper with the API key from environment variable
    comprehensive_answer = await scraper.scrape(query)
    logger.debug(f"Final comprehensive answer: {comprehensive_answer[:100]}...")  # Log first 100 characters
    return comprehensive_answer

# Example usage
if __name__ == "__main__":
    query = "What is the capital of France?"
    answer = asyncio.run(search_internet_and_process(query))
    print(f"Answer: {answer}")