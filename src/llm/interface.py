"""
Centralized LLM Interface for News RAG System

This module provides a unified interface for all LLM interactions across the system:
- Standardized API calls to the LLM server
- Automatic thinking block removal
- Structured output parsing (JSON arrays, objects, plain text)
- Consistent error handling and retry logic
- Centralized configuration management
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from ..config import get_config
except ImportError:
    # Handle direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_config

logger = logging.getLogger(__name__)


class LLMInterface:
    """Centralized interface for all LLM interactions"""
    
    def __init__(self, config=None):
        """Initialize the LLM interface"""
        if config is None:
            config = get_config()
            
        self.config = config
        self.llm_endpoint = config.get('llm', 'endpoint', default='http://localhost:8001/v1')
        self.llm_model = config.get('llm', 'model', default='qwen3-30b')
        
        # Default settings
        self.default_temperature = config.get('llm', 'temperature', default=0.3)
        self.default_max_tokens = config.get('llm', 'max_tokens', default=2000)
        self.max_retries = config.get('llm', 'max_retries', default=3)
        self.timeout = config.get('llm', 'timeout', default=60)
        
        # Initialize OpenAI client
        if OpenAI is None:
            logger.error("OpenAI package not found. Install with: pip install openai")
            raise ImportError("OpenAI package is required for LLM interface. Install with: pip install openai")
        
        self.client = OpenAI(
            base_url=self.llm_endpoint,
            api_key="not-needed"  # llama.cpp doesn't require API key
        )
        logger.info(f"Initialized LLM interface for {self.llm_endpoint}")
    
    def prompt_llm(self, 
                   prompt: str, 
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   output_type: str = "text",
                   n: int = 1,
                   model: Optional[str] = None,
                   strip_thinking: bool = True) -> Any:
        """
        Send a prompt to the LLM and return the processed response.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (0.0 to 1.0). Defaults to config value.
            max_tokens: Maximum tokens to generate. Defaults to config value.
            output_type: Expected output format:
                - "text": Plain text response
                - "json_array": JSON array (returns list)
                - "json_object": JSON object (returns dict)
                - "json_any": Any JSON structure
            n: Number of responses to generate (usually 1)
            model: LLM model to use. Defaults to config value.
            strip_thinking: Whether to remove thinking blocks from response
            
        Returns:
            Processed response based on output_type:
            - "text": str
            - "json_array": list
            - "json_object": dict
            - "json_any": dict or list
            
        Raises:
            LLMError: If the LLM call fails after retries
            JSONParseError: If JSON parsing fails for json output types
        """
        max_tokens=100000000
        # Use defaults if not specified
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if model is None:
            model = self.llm_model
        
        # Validate parameters
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if output_type not in ["text", "json_array", "json_object", "json_any"]:
            raise ValueError(f"Invalid output_type: {output_type}")
        
        logger.debug(f"LLM call: temp={temperature}, max_tokens={max_tokens}, output_type={output_type}")
        
        # Make the LLM call with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n
                )
                
                # Extract content
                raw_content = response.choices[0].message.content
                if not raw_content:
                    raise LLMError("LLM returned empty response")
                
                logger.debug(f"Raw LLM response length: {len(raw_content)}")
                
                # Process the response
                return self._process_response(raw_content, output_type, strip_thinking)
                
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise LLMError(f"LLM call failed after {self.max_retries} attempts: {e}")
                continue
    
    def _process_response(self, raw_content: str, output_type: str, strip_thinking: bool) -> Any:
        """Process the raw LLM response based on output type"""
        
        # Strip thinking blocks if requested
        if strip_thinking:
            content = self._strip_thinking_blocks(raw_content)
        else:
            content = raw_content
        
        # Process based on output type
        if output_type == "text":
            return content.strip()
        
        elif output_type == "json_array":
            return self._parse_json_array(content)
        
        elif output_type == "json_object":
            return self._parse_json_object(content)
        
        elif output_type == "json_any":
            return self._parse_json_any(content)
        
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")
    
    def _strip_thinking_blocks(self, content: str) -> str:
        """Remove thinking blocks from LLM response"""
        # Remove <think>...</think> blocks (case insensitive)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove <THINK>...</THINK> blocks
        content = re.sub(r'<THINK>.*?</THINK>', '', content, flags=re.DOTALL)
        
        return content.strip()
    
    def _parse_json_array(self, content: str) -> List[Any]:
        """Parse JSON array from content"""
        try:
            # First try to parse the entire content as JSON
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
            else:
                raise JSONParseError(f"Expected JSON array, got {type(parsed)}")
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON array in the response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if not json_match:
            logger.warning(f"No JSON array found in response: {content[:200]}...")
            return []
        
        json_str = json_match.group(0)
        
        # Clean common JSON formatting issues
        json_str = self._clean_json_string(json_str)
        
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return parsed
            else:
                raise JSONParseError(f"Expected JSON array, got {type(parsed)}")
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON array after cleaning: {e}")
            
            # Last resort: try to extract individual objects
            return self._extract_json_objects_from_array(json_str)
    
    def _parse_json_object(self, content: str) -> Dict[str, Any]:
        """Parse JSON object from content"""
        try:
            # First try to parse the entire content as JSON
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
            else:
                raise JSONParseError(f"Expected JSON object, got {type(parsed)}")
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if not json_match:
            logger.warning(f"No JSON object found in response: {content[:200]}...")
            return {}
        
        json_str = json_match.group(0)
        
        # Clean common JSON formatting issues
        json_str = self._clean_json_string(json_str)
        
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
            else:
                raise JSONParseError(f"Expected JSON object, got {type(parsed)}")
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON object after cleaning: {e}")
            return {}
    
    def _parse_json_any(self, content: str) -> Union[Dict[str, Any], List[Any]]:
        """Parse any JSON structure from content"""
        try:
            # First try to parse the entire content as JSON
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try JSON object first
        try:
            return self._parse_json_object(content)
        except JSONParseError:
            pass
        
        # Try JSON array
        try:
            return self._parse_json_array(content)
        except JSONParseError:
            pass
        
        logger.warning(f"Could not parse any JSON from response: {content[:200]}...")
        return {}
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean common JSON formatting issues"""
        fixes = [
            (r',\s*}', '}'),                    # Remove trailing commas in objects
            (r',\s*]', ']'),                    # Remove trailing commas in arrays
            (r'\\n', ' '),                      # Replace newlines in strings
            (r'\\"', '"'),                      # Fix escaped quotes
            (r'"\s*\n\s*"', '"'),              # Fix split strings across lines
        ]
        
        for pattern, replacement in fixes:
            json_str = re.sub(pattern, replacement, json_str)
        
        return json_str
    
    def _extract_json_objects_from_array(self, json_str: str) -> List[Dict[str, Any]]:
        """Last resort: extract individual JSON objects from a malformed array"""
        object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        objects = re.findall(object_pattern, json_str)
        
        parsed_objects = []
        for obj_str in objects:
            try:
                obj = json.loads(obj_str)
                if isinstance(obj, dict):
                    parsed_objects.append(obj)
            except json.JSONDecodeError:
                continue
        
        if parsed_objects:
            logger.info(f"Recovered {len(parsed_objects)} objects from malformed JSON array")
        
        return parsed_objects
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current LLM configuration"""
        return {
            'endpoint': self.llm_endpoint,
            'model': self.llm_model,
            'default_temperature': self.default_temperature,
            'default_max_tokens': self.default_max_tokens,
            'max_retries': self.max_retries,
            'timeout': self.timeout
        }


class LLMError(Exception):
    """Exception raised when LLM calls fail"""
    pass


class JSONParseError(Exception):
    """Exception raised when JSON parsing fails"""
    pass


# Convenience functions for common use cases
def prompt_for_text(prompt: str, temperature: float = 0.3, max_tokens: int = 2000, **kwargs) -> str:
    """Convenience function for text-only prompts"""
    interface = LLMInterface()
    return interface.prompt_llm(prompt, temperature=temperature, max_tokens=max_tokens, 
                               output_type="text", **kwargs)


def prompt_for_json_array(prompt: str, temperature: float = 0.1, max_tokens: int = 5000, **kwargs) -> List[Any]:
    """Convenience function for JSON array responses"""
    interface = LLMInterface()
    return interface.prompt_llm(prompt, temperature=temperature, max_tokens=max_tokens, 
                               output_type="json_array", **kwargs)


def prompt_for_json_object(prompt: str, temperature: float = 0.1, max_tokens: int = 5000, **kwargs) -> Dict[str, Any]:
    """Convenience function for JSON object responses"""
    interface = LLMInterface()
    return interface.prompt_llm(prompt, temperature=temperature, max_tokens=max_tokens, 
                               output_type="json_object", **kwargs)


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import init_config
    
    # Initialize
    config = init_config()
    llm = LLMInterface(config)
    
    # Test text response
    print("Testing text response...")
    text_result = llm.prompt_llm(
        "What is the capital of France?", 
        output_type="text",
        temperature=0.1,
        max_tokens=100
    )
    print(f"Text result: {text_result}")
    
    # Test JSON array response
    print("\nTesting JSON array response...")
    array_result = llm.prompt_llm(
        "List 3 major tech companies as a JSON array of strings.",
        output_type="json_array",
        temperature=0.1,
        max_tokens=200
    )
    print(f"Array result: {array_result}")
    
    # Test JSON object response
    print("\nTesting JSON object response...")
    object_result = llm.prompt_llm(
        'Return information about Apple Inc as JSON: {"name": "...", "founded": "...", "industry": "..."}',
        output_type="json_object",
        temperature=0.1,
        max_tokens=300
    )
    print(f"Object result: {object_result}")
    
    print(f"\nModel info: {llm.get_model_info()}")