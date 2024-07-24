import requests
from bs4 import BeautifulSoup
import logging
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

logging.basicConfig(level=logging.INFO)

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def scrape_url(self, url, source, num_results):
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if source == 'Google':
                results = []
                for i, g in enumerate(soup.find_all('div', class_='g')):
                    if i >= num_results:
                        break
                    anchor = g.find('a')
                    if anchor:
                        title = g.find('h3')
                        snippet = g.find('div', class_='VwiC3b')
                        if title and snippet:
                            results.append({
                                'source': source,
                                'content': f"{title.text}. {snippet.text}"
                            })
                return results
            
        except requests.RequestException as e:
            logging.error(f"Error scraping {url}: {e}")
        return None

    def scrape(self, query, num_results):
        logging.info(f"Scraping data for query: {query}")
        encoded_query = quote_plus(query)
        urls = [
            (f"https://www.google.com/search?q={encoded_query}&num={num_results}", 'Google'),
        ]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(lambda x: self.scrape_url(*x, num_results), urls))
        
        data = [item for sublist in results if sublist for item in sublist]
        logging.info(f"Scraped data: {data}")
        return data

if __name__ == "__main__":
    scraper = WebScraper()
    query = input("Enter your question: ")
    num_results = int(input("Enter the number of search results: "))
    results = scraper.scrape(query, num_results)
    for result in results:
        print(f"Source: {result['source']}")
        print(f"Content: {result['content'][:100]}...")  # Print first 100 characters
        print()
