import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
from datetime import datetime
from urllib.parse import quote_plus
from .custom_logger import CustomLogger

class WebScraper:
    """Provide clean web data for the model to analyze."""
    
    def __init__(self, verbose: bool = True):
        self.logger = CustomLogger.get_logger("WebScraper", verbose)
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        self.logger.info("WebScraper initialized with verbose logging", "WEB")

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        return ' '.join(text.split())

    def search(self, query: str) -> Optional[Dict]:
        """Get search results and let model interpret them."""
        try:
            self.logger.info(f"Processing search query: {query}", "WEB")
            search_results = []
            
            # Try Google Search first
            search_url = f"https://www.google.com/search?q={quote_plus(query)}"
            self.logger.debug(f"Trying Search URL: {search_url}", "WEB")
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                self.logger.success("Retrieved Search data", "WEB")
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Get featured snippet if available
                featured = soup.select_one('div.xpdopen, div[data-tts="answers"]')
                if featured:
                    search_results.append({
                        "type": "featured",
                        "content": self._clean_text(featured.get_text())
                    })
                
                # Get regular search results
                for result in soup.select('div.g'):
                    title = result.select_one('h3')
                    snippet = result.select_one('.VwiC3b')
                    if title and snippet:
                        search_results.append({
                            "type": "result",
                            "title": self._clean_text(title.get_text()),
                            "content": self._clean_text(snippet.get_text())
                        })
            
            # Try Google Maps for location-related queries
            if any(word in query.lower() for word in ["where", "location", "address", "distance", "route"]):
                maps_url = f"https://www.google.com/maps/search/{quote_plus(query)}"
                self.logger.debug(f"Trying Maps URL: {maps_url}", "WEB")
                
                response = requests.get(maps_url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    self.logger.success("Retrieved Maps data", "WEB")
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract location data
                    for item in soup.select('[role="article"], [data-item-id]'):
                        location_data = {}
                        
                        # Get address
                        address = item.select_one('[data-item-id="address"]')
                        if address:
                            location_data["address"] = self._clean_text(address.get_text())
                            
                        # Get distance/time info
                        distance = item.select_one('[jstcache] span[jstcache]')
                        if distance:
                            location_data["distance"] = self._clean_text(distance.get_text())
                            
                        if location_data:
                            search_results.append({
                                "type": "location",
                                "data": location_data
                            })
            
            if search_results:
                # Format data for model
                data = {
                    "system": "You have access to search results. Please analyze them and provide an accurate response.",
                    "context": (
                        "Search Results:\n\n" +
                        "\n\n".join([
                            f"[{result['type'].title()}]\n" + 
                            (f"Title: {result['title']}\n" if 'title' in result else "") +
                            (f"Content: {result['content']}" if 'content' in result else 
                             "\n".join(f"{k}: {v}" for k, v in result['data'].items()))
                            for result in search_results
                        ]) +
                        f"\n\nCurrent Time: {datetime.now().strftime('%B %d, %Y at %I:%M %p %Z')}"
                    ),
                    "query": query
                }
                self.logger.success("Search data retrieved and formatted", "WEB")
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Search error: {e}", "WEB")
            return None