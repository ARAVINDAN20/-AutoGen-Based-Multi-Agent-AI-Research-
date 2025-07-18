from typing import Dict, Any, List
import logging
from utils.model_manager import ModelManager

logger = logging.getLogger(__name__)

class DocumentationAgent:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.agent_name = "DocumentationAgent"
        self.model_type = "mistral"
        
        self.system_prompt = """
You are a Documentation Agent specialized in creating comprehensive, well-structured documentation.
Your capabilities include:
- Converting research findings into clear documentation
- Creating technical documentation with proper formatting
- Summarizing complex information into digestible content
- Organizing information with clear hierarchies and sections

Always provide:
1. Well-structured documents with clear headings
2. Proper formatting (markdown when appropriate)
3. Executive summaries for complex topics
4. Clear, concise language
5. Logical information flow
"""
    
    def format_prompt(self, task: str, content: str, doc_type: str = "general") -> str:
        """Format prompt for documentation generation"""
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"Documentation Task: {task}\n"
        prompt += f"Document Type: {doc_type}\n\n"
        prompt += f"Source Content:\n{content}\n\n"
        prompt += "Please create comprehensive documentation:"
        
        return prompt
    
    def create_documentation(self, research_data: Dict[str, Any], doc_type: str = "research_report") -> Dict[str, Any]:
        """Create documentation from research data"""
        try:
            logger.info(f"Creating {doc_type} documentation")
            
            # Extract content from research data
            if "research_report" in research_data:
                content = research_data["research_report"]
                sources = research_data.get("sources", [])
            else:
                content = str(research_data)
                sources = []
            
            # Create documentation prompt
            task = f"Create a {doc_type} document"
            prompt = self.format_prompt(task, content, doc_type)
            
            # Generate documentation
            documentation = self.model_manager.generate_response(
                model_type=self.model_type,
                prompt=prompt,
                max_tokens=1024,
                temperature=0.5
            )
            
            # Format final document
            final_doc = self._format_final_document(
                documentation, sources, doc_type
            )
            
            result = {
                "agent": self.agent_name,
                "document_type": doc_type,
                "documentation": final_doc,
                "source_research": research_data.get("query", ""),
                "sources_used": len(sources),
                "status": "completed"
            }
            
            logger.info(f"Documentation created successfully")
            return result
            
        except Exception as e:
            logger.error(f"Documentation creation failed: {str(e)}")
            return {
                "agent": self.agent_name,
                "error": str(e),
                "status": "failed"
            }
    
    def _format_final_document(self, content: str, sources: List[Dict], doc_type: str) -> str:
        """Format the final document with proper structure"""
        formatted_doc = f"# {doc_type.replace('_', ' ').title()}\n\n"
        formatted_doc += content
        
        if sources:
            formatted_doc += "\n\n## Sources\n\n"
            for i, source in enumerate(sources, 1):
                formatted_doc += f"{i}. [{source.get('title', 'Untitled')}]({source.get('url', '#')})\n"
        
        formatted_doc += f"\n\n---\n*Generated by {self.agent_name}*"
        
        return formatted_doc
    
    def summarize_content(self, content: str, summary_type: str = "executive") -> Dict[str, Any]:
        """Create different types of summaries"""
        try:
            prompt = f"{self.system_prompt}\n\n"
            prompt += f"Create a {summary_type} summary of the following content:\n\n"
            prompt += f"{content}\n\n"
            prompt += f"Provide a clear, concise {summary_type} summary:"
            
            summary = self.model_manager.generate_response(
                model_type=self.model_type,
                prompt=prompt,
                max_tokens=512,
                temperature=0.4
            )
            
            return {
                "agent": self.agent_name,
                "summary_type": summary_type,
                "summary": summary,
                "original_length": len(content),
                "summary_length": len(summary),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return {"error": str(e), "status": "failed"}

