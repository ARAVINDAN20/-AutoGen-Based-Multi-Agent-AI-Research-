from typing import Dict, Any, List
import logging
import re
from utils.model_manager import ModelManager

logger = logging.getLogger(__name__)

class CodingAgent:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.agent_name = "CodingAgent"
        self.model_type = "qwen"
        
        self.system_prompt = """
You are a Coding Agent specialized in software development and code generation.
Your capabilities include:
- Writing clean, efficient code in multiple programming languages
- Code review and optimization suggestions
- Debugging and error resolution
- Creating complete project structures
- Generating documentation for code

Always provide:
1. Clean, well-commented code
2. Proper error handling
3. Best practices and conventions
4. Performance considerations
5. Security considerations when applicable

Programming languages you excel at: Python, JavaScript, Java, C++, Go, Rust, and more.
"""
    
    def format_coding_prompt(self, task: str, language: str = "python", requirements: str = "") -> str:
        """Format prompt for coding tasks"""
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"Coding Task: {task}\n"
        prompt += f"Programming Language: {language}\n"
        
        if requirements:
            prompt += f"Requirements: {requirements}\n"
        
        prompt += "\nPlease provide the complete code solution with explanations:"
        
        return prompt
    
    def generate_code(self, task: str, language: str = "python", requirements: str = "") -> Dict[str, Any]:
        """Generate code based on requirements"""
        try:
            logger.info(f"Generating {language} code for: {task}")
            
            prompt = self.format_coding_prompt(task, language, requirements)
            
            code_response = self.model_manager.generate_response(
                model_type=self.model_type,
                prompt=prompt,
                max_tokens=1024,
                temperature=0.3
            )
            
            # Extract code blocks and explanations
            code_blocks = self._extract_code_blocks(code_response)
            explanation = self._extract_explanation(code_response)
            
            result = {
                "agent": self.agent_name,
                "task": task,
                "language": language,
                "code_blocks": code_blocks,
                "explanation": explanation,
                "full_response": code_response,
                "requirements": requirements,
                "status": "completed"
            }
            
            logger.info(f"Code generation completed for: {task}")
            return result
            
        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            return {
                "agent": self.agent_name,
                "task": task,
                "error": str(e),
                "status": "failed"
            }
    
    def review_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Review and provide suggestions for existing code"""
        try:
            prompt = f"{self.system_prompt}\n\n"
            prompt += f"Please review the following {language} code and provide:\n"
            prompt += "1. Code quality assessment\n"
            prompt += "2. Optimization suggestions\n"
            prompt += "3. Security considerations\n"
            prompt += "4. Best practices recommendations\n\n"
            prompt += f"Code to review:\n``````\n\n"
            prompt += "Provide detailed code review:"
            
            review = self.model_manager.generate_response(
                model_type=self.model_type,
                prompt=prompt,
                max_tokens=768,
                temperature=0.4
            )
            
            return {
                "agent": self.agent_name,
                "code_review": review,
                "language": language,
                "original_code": code,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Code review failed: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def debug_code(self, code: str, error_message: str, language: str = "python") -> Dict[str, Any]:
        """Debug code and provide fixes"""
        try:
            prompt = f"{self.system_prompt}\n\n"
            prompt += f"Debug the following {language} code that produces this error:\n"
            prompt += f"Error: {error_message}\n\n"
            prompt += f"Code:\n``````\n\n"
            prompt += "Please provide:\n1. Error explanation\n2. Fixed code\n3. Prevention suggestions"
            
            debug_response = self.model_manager.generate_response(
                model_type=self.model_type,
                prompt=prompt,
                max_tokens=768,
                temperature=0.3
            )
            
            return {
                "agent": self.agent_name,
                "debug_response": debug_response,
                "original_code": code,
                "error_message": error_message,
                "language": language,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Debugging failed: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def _extract_code_blocks(self, response: str) -> List[Dict[str, str]]:
        """Extract code blocks from response"""
        code_blocks = []
        pattern = r'``````'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for i, (language, code) in enumerate(matches):
            code_blocks.append({
                "id": i + 1,
                "language": language or "text",
                "code": code.strip()
            })
        
        return code_blocks
    
    def _extract_explanation(self, response: str) -> str:
        """Extract explanation text (non-code parts)"""
        # Remove code blocks and return remaining text
        explanation = re.sub(r'``````', '', response, flags=re.DOTALL)
        return explanation.strip()

