"""
Factory module for creating LLM instances with multi-cloud support.

This module implements the Factory pattern to abstract the creation of
different LLM strategies allowing for seamless switching between 
different models and deployment environments.
"""

from typing import Dict, Optional, Any, Type

from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.cloud_providers.cloud_provider_factory import CloudProviderFactory
from src.carmina.llm.strategies.anthropic_strategy import AnthropicStrategy
#from llm.strategies.openai_strategy import OpenAIStrategy
#from llm.strategies.huggingface_strategy import HuggingFaceStrategy
#from llm.strategies.mock_strategy import MockStrategy

class LLMFactory:
    """Factory for creating LLM strategy instances with cloud provider integration."""

    # Mapping of model name patterns to strategy classes
    _strategies: Dict[str, Type[BaseLLMStrategy]] = {
        # "gpt": OpenAIStrategy,
        "claude": AnthropicStrategy,
        "anthropic": AnthropicStrategy,
        # "llama": HuggingFaceStrategy,
        # "mistral": HuggingFaceStrategy,
        # "mock": MockStrategy,
    }

    @classmethod
    def create(
        cls, 
        model_name: str, 
        cloud_provider: str, 
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        provider_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BaseLLMStrategy:
        """
        Create an LLM strategy instance configured for the specified cloud provider.

        Args:
            model_name: Name of the model to use.
            cloud_provider: Name of the cloud provider to use.
            strategy_kwargs: Additional configuration for the LLM strategy.
            provider_kwargs: Additional configuration for the cloud provider.

        Returns:
            An initialized LLM strategy instance.

        Raises:
            ValueError: If no suitable strategy is found for the model name.
        """
        strategy_kwargs = strategy_kwargs or {}
        provider_kwargs = provider_kwargs or {}
        
        # Initialize the cloud provider
        provider = CloudProviderFactory.create(cloud_provider, **provider_kwargs)
        
        # Find the appropriate strategy for the model name
        strategy_class = cls._get_strategy_for_model(model_name)
        
        # Create and return the configured strategy
        return strategy_class(
            model_name=model_name,
            cloud_provider=provider,
            **strategy_kwargs
        )

    @classmethod
    def _get_strategy_for_model(cls, model_name: str) -> Type[BaseLLMStrategy]:
        """
        Find the appropriate strategy class for the given model name.

        Args:
            model_name: Name of the model to find a strategy for.

        Returns:
            The strategy class to use.

        Raises:
            ValueError: If no suitable strategy is found.
        """
        model_name = model_name.lower()
        
        for pattern, strategy_class in cls._strategies.items():
            if pattern in model_name:
                return strategy_class
                
        raise ValueError(
            f"No suitable strategy found for model: {model_name}. "
            f"Supported model patterns are: {', '.join(cls._strategies.keys())}"
        )

    @classmethod
    def register_strategy(cls, pattern: str, strategy_class: Type[BaseLLMStrategy]) -> None:
        """
        Register a new strategy class for a specific model name pattern.

        Args:
            pattern: The model name pattern to match.
            strategy_class: The strategy class to use for matching models.
        """
        cls._strategies[pattern.lower()] = strategy_class