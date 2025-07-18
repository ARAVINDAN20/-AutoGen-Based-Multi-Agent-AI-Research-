import asyncio
import logging
from typing import Dict, Any, List, Optional
from agents.research.research_agent import ResearchAgent
from agents.documentation.documentation_agent import DocumentationAgent
from agents.coding.coding_agent import CodingAgent
from utils.model_manager import ModelManager

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(self):
        self.model_manager = ModelManager()
        self.agents = {}
        self.conversation_history = []
        self.active_agents = []
        
    async def initialize_agents(self):
        """Initialize all agents with their respective models"""
        try:
            logger.info("Initializing Agent Orchestrator...")
            
            # Load open-source models (no gated repos)
            models_to_load = [
                ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
                ("microsoft/DialoGPT-medium", "mistral"),  # Alternative to gated Mistral
                ("Qwen/Qwen2.5-7B-Instruct", "qwen")
            ]
            
            loaded_count = 0
            for model_name, model_type in models_to_load:
                try:
                    success = self.model_manager.load_model(model_name, model_type)
                    if success:
                        loaded_count += 1
                    else:
                        logger.warning(f"Failed to load {model_name}")
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {str(e)}")
            
            if loaded_count == 0:
                logger.error("No models loaded successfully")
                return False
            
            # Initialize agents
            self.agents = {
                "research": ResearchAgent(self.model_manager),
                "documentation": DocumentationAgent(self.model_manager),
                "coding": CodingAgent(self.model_manager)
            }
            
            logger.info(f"Agent Orchestrator initialized with {loaded_count} models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            return False
    
    async def process_user_request(self, request: str, selected_agents: List[str] = None) -> Dict[str, Any]:
        """Process user request with selected agents"""
        try:
            if not selected_agents:
                selected_agents = self._determine_agents_needed(request)
            
            logger.info(f"Processing request with agents: {selected_agents}")
            
            results = {}
            conversation_id = len(self.conversation_history)
            
            # Execute agents in sequence
            if "research" in selected_agents:
                try:
                    research_result = await self.agents["research"].conduct_research(request)
                    results["research"] = research_result
                except Exception as e:
                    logger.error(f"Research agent failed: {str(e)}")
                    results["research"] = {"error": str(e), "status": "failed"}
            
            if "documentation" in selected_agents:
                try:
                    research_data = results.get("research", {"research_report": request})
                    doc_result = self.agents["documentation"].create_documentation(research_data)
                    results["documentation"] = doc_result
                except Exception as e:
                    logger.error(f"Documentation agent failed: {str(e)}")
                    results["documentation"] = {"error": str(e), "status": "failed"}
            
            if "coding" in selected_agents:
                try:
                    if self._is_coding_request(request):
                        code_result = self.agents["coding"].generate_code(request)
                        results["coding"] = code_result
                except Exception as e:
                    logger.error(f"Coding agent failed: {str(e)}")
                    results["coding"] = {"error": str(e), "status": "failed"}
            
            # Store conversation
            conversation_entry = {
                "id": conversation_id,
                "request": request,
                "selected_agents": selected_agents,
                "results": results,
                "timestamp": asyncio.get_event_loop().time()
            }
            self.conversation_history.append(conversation_entry)
            
            return {
                "conversation_id": conversation_id,
                "request": request,
                "results": results,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Request processing failed: {str(e)}")
            return {
                "request": request,
                "error": str(e),
                "status": "failed"
            }
    
    def _determine_agents_needed(self, request: str) -> List[str]:
        """Determine which agents are needed based on request content"""
        request_lower = request.lower()
        needed_agents = []
        
        # Research keywords
        research_keywords = ["research", "find", "search", "information", "analyze", "study"]
        if any(keyword in request_lower for keyword in research_keywords):
            needed_agents.append("research")
        
        # Documentation keywords
        doc_keywords = ["document", "write", "report", "summary", "documentation"]
        if any(keyword in request_lower for keyword in doc_keywords):
            needed_agents.append("documentation")
        
        # Coding keywords
        code_keywords = ["code", "program", "script", "function", "class", "algorithm", "debug"]
        if any(keyword in request_lower for keyword in code_keywords):
            needed_agents.append("coding")
        
        # Default to research if no specific keywords found
        if not needed_agents:
            needed_agents.append("research")
        
        return needed_agents
    
    def _is_coding_request(self, request: str) -> bool:
        """Check if request is specifically about coding"""
        coding_indicators = [
            "write code", "create function", "build script", "program",
            "algorithm", "class", "method", "debug", "fix code"
        ]
        return any(indicator in request.lower() for indicator in coding_indicators)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents and models"""
        return {
            "agents": list(self.agents.keys()),
            "model_info": self.model_manager.get_model_info(),
            "conversation_count": len(self.conversation_history),
            "active_agents": self.active_agents
        }
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]

