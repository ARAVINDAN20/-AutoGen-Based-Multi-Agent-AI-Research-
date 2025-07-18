from duckduckgo_search import DDGS
from typing import List, Dict, Any
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class WebSearchManager:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def search_web(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Perform web search using DuckDuckGo"""
        try:
            max_results = max_results or self.max_results
            
            logger.info(f"Searching for: {query}")
            
            results = []
            # Use DDGS without proxy parameters to avoid the error
            ddgs = DDGS()
            search_results = ddgs.text(
                keywords=query,
                max_results=max_results,
                safesearch='moderate'
            )
                
            for result in search_results:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', ''),
                    'source': 'DuckDuckGo'
                })
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
