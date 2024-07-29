#!/usr/bin/env python3

import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "facebook/bart-large-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)

    def scrape(self, query, num_results=5):
        search_url = f"https://www.google.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            search_results = []
            for g in soup.find_all('div', class_='g'):
                anchor = g.find('a')
                if anchor and 'href' in anchor.attrs:
                    link = anchor['href']
                    if link.startswith('/url?q='):
                        link = link.split('/url?q=')[1].split('&')[0]
                    if self.is_valid_url(link):
                        title = g.find('h3', class_='r')
                        title = title.text if title else "No title"
                        snippet = g.find('div', class_='s')
                        snippet = snippet.text if snippet else "No snippet"
                        search_results.append({
                            'title': title,
                            'link': link,
                            'snippet': snippet
                        })
                if len(search_results) >= num_results:
                    break

            scraped_data = []
            formatted_data = []
            for result in search_results:
                try:
                    page_content = self.fetch_page_content(result['link'])
                    relevance_score = self.check_relevance(query, page_content)
                    if relevance_score > 0.5:  # Adjust this threshold as needed
                        scraped_data.append({
                            'title': result['title'],
                            'content': page_content,
                            'url': result['link']
                        })
                        formatted_data.append(f"Title: {result['title']}\nURL: {result['link']}\nContent: {page_content[:500]}...\n")
                except Exception as e:
                    logger.error(f"Error scraping {result['link']}: {str(e)}")

            return scraped_data, formatted_data

        except requests.RequestException as e:
            logger.error(f"Error during web scraping: {str(e)}")
            return [], []

    def is_valid_url(self, url):
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    def fetch_page_content(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return ' '.join(p.get_text() for p in soup.find_all('p'))
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return ""

    def check_relevance(self, query, content):
        inputs = self.tokenizer(query, content, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        return scores[:, 1].item()  # Return the probability of entailment (relevance)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Web Scraper Script")
    parser.add_argument("--query", "-q", type=str, required=True, help="The query to search for")
    parser.add_argument("--results", "-r", type=int, default=5, help="Number of results to retrieve")
    args = parser.parse_args()

    scraper = WebScraper()
    scraped_data, formatted_data = scraper.scrape(args.query, args.results)

    print(f"Scraped {len(scraped_data)} results:")
    for item in formatted_data:
        print(item)
        print("-" * 80)