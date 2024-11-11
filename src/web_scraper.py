import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import json
import os
from urllib.parse import quote_plus
import sys
import time
import yfinance as yf  # Add Yahoo Finance for real-time stock data

class WebScraper:
    """Simple web scraper that lets model handle search results naturally."""
    
    def __init__(self, cache_dir: str = "data/web_cache"):
        """Initialize with caching for efficiency."""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Common search engines with fallbacks
        self.search_urls = {
            "duckduckgo": "https://html.duckduckgo.com/html/?q={}",
            "brave": "https://search.brave.com/search?q={}",
            "google": "https://www.google.com/search?q={}"
        }
        
        # Headers to mimic browser
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
        
        # Cache settings - shorter for stock data
        self.cache_duration = 300  # 5 minutes for regular searches
        self.stock_cache_duration = 60  # 1 minute for stock data

    def get_stock_price(self, symbol: str) -> Optional[Dict]:
        """Get real-time stock data."""
        try:
            # Check cache first
            cache_file = os.path.join(
                self.cache_dir, 
                f"stock_{symbol.lower()}.json"
            )
            
            # Use cache if very fresh
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if time.time() - data.get("timestamp", 0) < self.stock_cache_duration:
                        return data
            
            # Get real-time data from Yahoo Finance
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Format stock data
            data = {
                "symbol": symbol,
                "price": info.get("regularMarketPrice"),
                "change": info.get("regularMarketChange"),
                "change_percent": info.get("regularMarketChangePercent"),
                "volume": info.get("regularMarketVolume"),
                "timestamp": time.time(),
                "market_time": info.get("regularMarketTime")
            }
            
            # Cache results
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return data
            
        except Exception as e:
            print(f"Error getting stock price: {e}", file=sys.stderr)
            return None

    def search(self, query: str, max_results: int = 3) -> str:
        """Search web and let model process results naturally."""
        try:
            # Check for stock price queries
            if any(word in query.lower() for word in ["stock", "price", "share"]):
                for symbol in ["NVDA", "AMD", "INTC", "TSLA", "AAPL", "MSFT", "GOOG"]:
                    if symbol.lower() in query.lower():
                        stock_data = self.get_stock_price(symbol)
                        if stock_data:
                            return (
                                f"Real-time Stock Data for {symbol}:\n"
                                f"Current Price: ${stock_data['price']:.2f}\n"
                                f"Change: {stock_data['change']:.2f} ({stock_data['change_percent']:.2f}%)\n"
                                f"Volume: {stock_data['volume']:,}\n"
                                f"Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stock_data['market_time']))}"
                            )
            
            # Regular web search if not stock query
            cache_file = os.path.join(
                self.cache_dir, 
                f"{quote_plus(query)}.json"
            )
            
            # Use cache if fresh
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if time.time() - data.get("timestamp", 0) < self.cache_duration:
                        return data["content"]
            
            # Try search engines in order
            results = []
            for engine, url in self.search_urls.items():
                try:
                    search_url = url.format(quote_plus(query))
                    response = requests.get(search_url, headers=self.headers, timeout=5)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Handle different search engine formats
                        if engine == "duckduckgo":
                            results = self._parse_duckduckgo(soup, max_results)
                        elif engine == "brave":
                            results = self._parse_brave(soup, max_results)
                        else:
                            results = self._parse_google(soup, max_results)
                            
                        if results:
                            break
                            
                except Exception as e:
                    print(f"Error with {engine}: {e}", file=sys.stderr)
                    continue
            
            if not results:
                return ""
                
            # Format results for model following NVIDIA's context format
            content = "Search Results:\n\n" + "\n\n".join([
                f"Source: {r.get('source', 'Unknown')}\n"
                f"Title: {r['title']}\n"
                f"Summary: {r['snippet']}"
                for r in results
            ])
            
            # Cache results with timestamp
            with open(cache_file, 'w') as f:
                json.dump({
                    "query": query,
                    "content": content,
                    "results": results,
                    "timestamp": time.time()
                }, f, indent=2)
            
            return content
            
        except Exception as e:
            print(f"Error searching web: {e}", file=sys.stderr)
            return ""

    def _parse_duckduckgo(self, soup: BeautifulSoup, max_results: int) -> List[Dict]:
        """Parse DuckDuckGo search results."""
        results = []
        for result in soup.select(".result"):
            title = result.select_one(".result__title")
            snippet = result.select_one(".result__snippet")
            source = result.select_one(".result__url")
            
            if title and snippet:
                results.append({
                    "title": title.get_text(strip=True),
                    "snippet": snippet.get_text(strip=True),
                    "source": source.get_text(strip=True) if source else "DuckDuckGo"
                })
                
            if len(results) >= max_results:
                break
        return results

    def _parse_brave(self, soup: BeautifulSoup, max_results: int) -> List[Dict]:
        """Parse Brave search results."""
        results = []
        for result in soup.select(".snippet"):
            title = result.select_one(".title")
            snippet = result.select_one(".snippet-description")
            source = result.select_one(".url")
            
            if title and snippet:
                results.append({
                    "title": title.get_text(strip=True),
                    "snippet": snippet.get_text(strip=True),
                    "source": source.get_text(strip=True) if source else "Brave Search"
                })
                
            if len(results) >= max_results:
                break
        return results

    def _parse_google(self, soup: BeautifulSoup, max_results: int) -> List[Dict]:
        """Parse Google search results."""
        results = []
        for result in soup.select(".g"):
            title = result.select_one("h3")
            snippet = result.select_one(".VwiC3b")
            source = result.select_one(".iUh30")
            
            if title and snippet:
                results.append({
                    "title": title.get_text(strip=True),
                    "snippet": snippet.get_text(strip=True),
                    "source": source.get_text(strip=True) if source else "Google"
                })
                
            if len(results) >= max_results:
                break
        return results

    def get_context(self, query: str) -> str:
        """Get search results as context for model."""
        results = self.search(query)
        if results:
            # Format following NVIDIA's context format
            return f"\nContext:\n{results}\n"
        return ""