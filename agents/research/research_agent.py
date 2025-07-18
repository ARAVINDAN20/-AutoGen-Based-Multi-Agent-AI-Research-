from typing import Dict, Any, List
import asyncio
import logging
from utils.model_manager import ModelManager
from utils.search import WebSearchManager

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.search_manager = WebSearchManager()
        self.agent_name = "ResearchAgent"
        self.model_type = "phi3"
        
        # System prompt for research tasks
        self.system_prompt = """
You are a Research Agent specialized in conducting comprehensive research on any topic.
Your capabilities include:
- Web searching and information gathering
- Analyzing and synthesizing information from multiple sources
- Providing detailed, well-structured research reports
- Fact-checking and verifying information accuracy

Always provide:
1. Clear, factual information
2. Multiple perspectives when relevant
3. Source citations
4. Structured, easy-to-read reports
"""
    
    def format_prompt(self, query: str, search_results: str = "") -> str:
        """Format the complete prompt for the model"""
        prompt = f"{self.system_prompt}\n\n"
        
        if search_results:
            prompt += f"Based on the following search results:\n{search_results}\n\n"
        
        prompt += f"Research Query: {query}\n\n"
        prompt += "Please provide a comprehensive research report:"
        
        return prompt
    
    async def conduct_research(self, query: str) -> Dict[str, Any]:
        """Conduct comprehensive research on a topic"""
        try:
            logger.info(f"Starting research on: {query}")
            
            # Step 1: Perform web search
            search_results = await self.search_manager.async_search(query, max_results=8)
            formatted_results = self.search_manager.format_search_results(search_results)
            
            # Step 2: Generate research report
            prompt = self.format_prompt(query, formatted_results)
            
            research_report = self.model_manager.generate_response(
                model_type=self.model_type,
                prompt=prompt,
                max_tokens=1024,
                temperature=0.7
            )
            
            # Step 3: Compile final research result
            result = {
                "agent": self.agent_name,
                "query": query,
                "research_report": research_report,
                "sources": search_results,
                "source_count": len(search_results),
                "status": "completed"
            }
            
            logger.info(f"Research completed for: {query}")
            return result
            
        except Exception as e:
            logger.error(f"Research failed: {str(e)}")
            return {
                "agent": self.agent_name,
                "query": query,
                "error": str(e),
                "status": "failed"
            }
    
    def search_specific_topic(self, topic: str, subtopics: List[str] = None) -> Dict[str, Any]:
        """Search for specific topic with optional subtopics"""
        try:
            all_results = []
            
            # Main topic search
            main_results = self.search_manager.search_web(topic)
            all_results.extend(main_results)
            
            # Subtopic searches
            if subtopics:
                for subtopic in subtopics:
                    sub_query = f"{topic} {subtopic}"
                    sub_results = self.search_manager.search_web(sub_query, max_results=3)
                    all_results.extend(sub_results)
            
            return {
                "topic": topic,
                "subtopics": subtopics or [],
                "results": all_results,
                "total_sources": len(all_results)
            }
            
        except Exception as e:
            logger.error(f"Specific topic search failed: {str(e)}")
            return {"error": str(e)}

