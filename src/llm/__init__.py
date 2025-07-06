"""
LLM Interface Module for News RAG System

This module provides centralized LLM interaction functionality.
"""

from .interface import (
    LLMInterface,
    LLMError,
    JSONParseError,
    prompt_for_text,
    prompt_for_json_array,
    prompt_for_json_object
)

__all__ = [
    'LLMInterface',
    'LLMError', 
    'JSONParseError',
    'prompt_for_text',
    'prompt_for_json_array',
    'prompt_for_json_object'
]