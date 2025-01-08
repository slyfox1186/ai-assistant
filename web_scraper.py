#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from custom_logger import CustomLogger, log_execution_time
import json
from urllib.parse import quote_plus
import time
import random
import re
import urllib.parse

logger = CustomLogger.get_logger()

class WebScraper:
    def __init__(self):
        """Initialize web scraper"""
        self.base_url = "https://www.google.com/search"
        self.maps_url = "https://www.google.com/maps/search/"
        self.directions_url = "https://www.google.com/maps/dir/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        logger.info("WebScraper initialized successfully")

    @log_execution_time
    def search_google(self, query: str) -> List[Dict[str, str]]:
        """Perform a Google search using direct URL"""
        try:
            logger.debug(f"Starting search with query: {query}")
            
            params = {
                'client': 'ubuntu-sn',
                'channel': 'fs',
                'q': quote_plus(query)
            }
            
            logger.debug(f"Request URL: {self.base_url}")
            logger.debug(f"Request params: {params}")
            
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            logger.debug(f"Raw response text: {response.text[:1000]}...")  # First 1000 chars
            
            # Parse results
            soup = BeautifulSoup(response.text, 'html.parser')
            search_results = []
            
            # Log what we find
            for result in soup.find_all('div', class_='g'):
                logger.debug(f"Found result div: {result}")
                try:
                    title_elem = result.find('h3')
                    link_elem = result.find('a')
                    desc_elem = result.find('div', class_='VwiC3b')
                    
                    if title_elem and link_elem and desc_elem:
                        result_data = {
                            'title': title_elem.get_text(),
                            'url': link_elem['href'],
                            'description': desc_elem.get_text()
                        }
                        logger.debug(f"Parsed result: {result_data}")
                        search_results.append(result_data)
                except Exception as e:
                    logger.warning(f"Failed to parse result: {str(e)}")
                    continue
            
            logger.debug(f"Final formatted results: {search_results}")
            return search_results
            
        except Exception as e:
            logger.error(f"Google search failed: {str(e)}")
            return []

    def search_maps(self, query: str) -> List[Dict[str, str]]:
        """Search Google Maps directly"""
        try:
            maps_url = f"{self.maps_url}{quote_plus(query)}"
            response = requests.get(
                maps_url,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for address and location info
            results = []
            
            # Maps typically shows address in meta tags
            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc:
                results.append({
                    'title': 'Google Maps Location',
                    'url': response.url,
                    'description': meta_desc.get('content', '')
                })
            
            return results
            
        except Exception as e:
            logger.warning(f"Maps search failed: {str(e)}")
            return []

    def format_search_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results for the model"""
        if not results:
            return "No search results found."
            
        formatted = ""
        for i, result in enumerate(results, 1):
            if result.get('map_view'):
                formatted += f"🗺️ {result['title']}\n"
                formatted += f"Route: {result['description']}\n"
                if result.get('route_info'):
                    formatted += "Directions:\n"
                    for step in result['route_info']:
                        formatted += f"- {step}\n"
                formatted += f"\nInteractive Map: {result['url']}\n"
            else:
                formatted += f"{result['description']}\n\n"
                if result['url'] and result['title']:
                    formatted += f"{i}. [{result['title']}]({result['url']})\n\n"
        
        return formatted.strip()

    def get_directions(self, query: str) -> Optional[str]:
        """Get directions between two locations"""
        try:
            # Add debug logging
            logger.debug(f"Getting directions for query: {query}")
            
            if "from" in query.lower() and "to" in query.lower():
                parts = query.lower().split("from")[1].split("to")
                origin = parts[0].strip()
                destination = parts[1].strip()
                
                # Log the extracted addresses
                logger.debug(f"Origin: {origin}")
                logger.debug(f"Destination: {destination}")
                
                url = f"https://www.google.com/maps/dir/{quote_plus(origin)}/{quote_plus(destination)}"
                logger.debug(f"Generated URL: {url}")
                
                # Verify URL is accessible
                response = requests.head(url)
                if response.status_code == 200:
                    return url
                else:
                    logger.error(f"URL check failed with status code: {response.status_code}")
                    return None
                
            return None
        except Exception as e:
            logger.error(f"Failed to get directions: {str(e)}")
            return None 