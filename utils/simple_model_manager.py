import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

logger = logging.getLogger(__name__)

class SimpleModelManager:
    def __init__(self):
        self.pipelines = {}
        
    def load_model(self, model_name: str, model_type: str) -> bool:
        """Load model without quantization to avoid accelerate conflicts"""
        try:
            logger.info(f"Loading {model_name} (simple mode)")
            
            # Load without quantization
            pipe = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device_map="auto",  # Let accelerate handle this
                trust_remote_code=True
                # Removed token argument; authentication handled via environment variable or huggingface-cli login
            )
            
            self.pipelines[model_type] = pipe
            logger.info(f"Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {str(e)}")
            return False
    
    def generate_response(self, model_type: str, prompt: str, **kwargs) -> str:
        """Generate response using specified model"""
        if model_type not in self.pipelines:
            return f"Model {model_type} not available"
        
        try:
            pipe = self.pipelines[model_type]
            result = pipe(prompt, max_new_tokens=kwargs.get('max_tokens', 512))
            # Handle different output keys robustly
            if 'generated_text' in result[0]:
                return result[0]['generated_text'][len(prompt):].strip()
            elif 'text' in result[0]:
                return result[0]['text'][len(prompt):].strip()
            else:
                return str(result[0])
        except Exception as e:
            return f"Error: {str(e)}"
