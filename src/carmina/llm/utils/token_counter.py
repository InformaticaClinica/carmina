"""
Token counting utilities for different LLM providers.

This module provides token counting functionality for various LLM models
to enable accurate cost tracking and usage monitoring.
"""

import re
from typing import Dict, List, Union, Optional
from abc import ABC, abstractmethod

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class BaseTokenCounter(ABC):
    """Base class for token counters."""
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        pass
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Count tokens in a list of messages (system + user prompts).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Dictionary with token counts for each role and total
        """
        counts = {"system": 0, "user": 0, "total": 0}
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            tokens = self.count_tokens(content)
            
            if role in counts:
                counts[role] += tokens
            counts["total"] += tokens
            
        return counts


class OpenAITokenCounter(BaseTokenCounter):
    """Token counter for OpenAI models using tiktoken."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is required for OpenAI token counting")
        
        # Map model names to tiktoken encodings
        model_encodings = {
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-4": "cl100k_base", 
            "gpt-4-turbo": "cl100k_base",
            "text-embedding-ada-002": "cl100k_base"
        }
        
        encoding_name = model_encodings.get(model_name, "cl100k_base")
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken encoding."""
        if not text:
            return 0
        return len(self.encoding.encode(text))


class AnthropicTokenCounter(BaseTokenCounter):
    """Token counter for Anthropic models using character-based approximation."""
    
    def __init__(self, model_name: str = "claude-3-sonnet"):
        self.model_name = model_name
        # Anthropic uses approximately 3.5-4 characters per token
        self.chars_per_token = 3.7
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using character-based approximation."""
        if not text:
            return 0
        return int(len(text) / self.chars_per_token)


class GenericTokenCounter(BaseTokenCounter):
    """Generic token counter using word-based approximation."""
    
    def __init__(self, tokens_per_word: float = 1.3):
        # Most languages average about 1.3 tokens per word
        self.tokens_per_word = tokens_per_word
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using word-based approximation."""
        if not text:
            return 0
        
        # Count words using regex to handle punctuation
        words = re.findall(r'\b\w+\b', text)
        return int(len(words) * self.tokens_per_word)


class MockTokenCounter(BaseTokenCounter):
    """Mock token counter for testing purposes."""
    
    def __init__(self, tokens_per_char: float = 0.25):
        # Simple approximation: 4 characters per token
        self.tokens_per_char = tokens_per_char
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using simple character-based formula."""
        if not text:
            return 0
        return int(len(text) * self.tokens_per_char)


def get_token_counter(model_name: str, provider_name: str = "unknown") -> BaseTokenCounter:
    """
    Get appropriate token counter for the given model and provider.
    
    Args:
        model_name: Name of the model (e.g., "gpt-4", "claude-3-sonnet")
        provider_name: Name of the provider (e.g., "openai", "anthropic")
        
    Returns:
        Appropriate token counter instance
    """
    model_lower = model_name.lower()
    provider_lower = provider_name.lower()
    
    # OpenAI models
    if "gpt" in model_lower or provider_lower == "openai":
        if TIKTOKEN_AVAILABLE:
            return OpenAITokenCounter(model_name)
        else:
            return GenericTokenCounter()
    
    # Anthropic models
    elif "claude" in model_lower or provider_lower == "anthropic":
        return AnthropicTokenCounter(model_name)
    
    # Mock models
    elif "mock" in model_lower or provider_lower == "mock":
        return MockTokenCounter()
    
    # Default to generic counter for other models
    else:
        return GenericTokenCounter()


def count_prompt_tokens(system_prompt: str, user_prompt: str, 
                       model_name: str = "unknown", 
                       provider_name: str = "unknown") -> Dict[str, int]:
    """
    Count tokens for both system and user prompts.
    
    Args:
        system_prompt: The system prompt text
        user_prompt: The user prompt text
        model_name: Name of the model
        provider_name: Name of the provider
        
    Returns:
        Dictionary with token counts for system, user, and total
    """
    counter = get_token_counter(model_name, provider_name)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return counter.count_message_tokens(messages)