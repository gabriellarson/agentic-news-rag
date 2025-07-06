"""
Configuration management for the News RAG System
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the News RAG system"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default config path relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "search_config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Override with environment variables if they exist
            config = self._apply_env_overrides(config)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # Common environment variable mappings
        env_mappings = {
            'NEWS_RAG_LLM_ENDPOINT': ['llm', 'endpoint'],
            'NEWS_RAG_LLM_MODEL': ['llm', 'model'],
            'NEWS_RAG_QDRANT_HOST': ['qdrant', 'host'],
            'NEWS_RAG_QDRANT_PORT': ['qdrant', 'port'],
            'NEWS_RAG_EMBEDDING_DEVICE': ['embeddings', 'device'],
            'NEWS_RAG_API_PORT': ['api', 'port'],
            'NEWS_RAG_LOG_LEVEL': ['logging', 'level'],
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Navigate to the config section and set the value
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Convert to appropriate type
                if env_value.lower() in ['true', 'false']:
                    env_value = env_value.lower() == 'true'
                elif env_value.isdigit():
                    env_value = int(env_value)
                elif env_value.replace('.', '').isdigit():
                    env_value = float(env_value)
                
                current[config_path[-1]] = env_value
                logger.info(f"Applied environment override: {env_var} = {env_value}")
        
        return config
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        current = self._config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.get('llm', default={})
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration"""
        return self.get('embeddings', default={})
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """Get Qdrant configuration"""
        return self.get('qdrant', default={})
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration"""
        return self.get('search', default={})
    
    def get_query_analysis_config(self) -> Dict[str, Any]:
        """Get query analysis configuration"""
        return self.get('query_analysis', default={})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.get('api', default={})
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.get('logging', default={})
        
        # Create logs directory if it doesn't exist
        logs_dir = Path(self.get('paths', 'logs_dir', default='logs/'))
        logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.StreamHandler(),  # Console output
                logging.FileHandler(
                    logs_dir / log_config.get('file', 'news_rag.log').split('/')[-1]
                )
            ]
        )
    
    def create_directories(self):
        """Create necessary directories based on configuration"""
        paths = self.get('paths', default={})
        
        for dir_key, dir_path in paths.items():
            if dir_key.endswith('_dir'):
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
    
    def validate_config(self) -> bool:
        """Validate configuration completeness"""
        required_sections = ['llm', 'embeddings', 'qdrant', 'search']
        
        for section in required_sections:
            if section not in self._config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate LLM config
        llm_config = self.get_llm_config()
        if not llm_config.get('endpoint') or not llm_config.get('model'):
            logger.error("LLM configuration missing endpoint or model")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def reload(self):
        """Reload configuration from file"""
        self._config = self._load_config()
        logger.info("Configuration reloaded")


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance"""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance


def init_config(config_path: Optional[str] = None) -> Config:
    """Initialize configuration and setup logging"""
    config = get_config(config_path)
    config.setup_logging()
    config.create_directories()
    
    if not config.validate_config():
        raise ValueError("Configuration validation failed")
    
    return config


# Example usage
if __name__ == "__main__":
    # Test configuration loading
    config = init_config()
    
    print("LLM Config:", config.get_llm_config())
    print("Search Alpha:", config.get('search', 'hybrid', 'alpha'))
    print("Articles Directory:", config.get('paths', 'articles_dir'))