"""
Factory module for creating LLM instances with multi-cloud support.

This module implements the Factory pattern to abstract the creation of
different cloud provider configurations, allowing 
for seamless switching between different models and deployment environments.
"""
from typing import Dict, Optional, Any, Type

#from src.carmina.llm.strategies.anthropic_strategy import AnthropicStrategy
from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.cloud_providers.aws_provider import AWSProvider
from src.carmina.llm.cloud_providers.azure_provider import AzureProvider
from src.carmina.llm.cloud_providers.google_ai_studio_provider import GoogleAIStudioProvider
from src.carmina.llm.cloud_providers.openai_provider import OpenAIProvider
from src.carmina.llm.cloud_providers.local_provider import LocalProvider
# from llm.strategies.huggingface_strategy import HuggingFaceStrategy
# from llm.strategies.mock_strategy import MockStrategy


class CloudProviderFactory:
    """Factory for creating cloud provider instances."""

    _providers: Dict[str, Type[BaseCloudProvider]] = {
        "aws": AWSProvider,
        "azure": AzureProvider,
        "google_ai_studio": GoogleAIStudioProvider,
        "openai": OpenAIProvider,
        "local": LocalProvider
    }

    @classmethod
    def create(cls, provider_name: str, **kwargs) -> BaseCloudProvider:
        """
        Create a cloud provider instance.

        Args:
            provider_name: Name of the cloud provider to create.
            **kwargs: Additional provider-specific configurations.

        Returns:
            An initialized cloud provider instance.

        Raises:
            ValueError: If the specified provider is not supported.
        """
        provider_name = provider_name.lower()
        if provider_name not in cls._providers:
            raise ValueError(
                f"Unsupported cloud provider: {provider_name}. "
                f"Supported providers are: {', '.join(cls._providers.keys())}"
            )
        
        provider_class = cls._providers[provider_name]
        return provider_class(**kwargs)

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseCloudProvider]) -> None:
        """
        Register a new cloud provider class.

        Args:
            name: Provider name identifier.
            provider_class: The provider class to register.
        """
        cls._providers[name.lower()] = provider_class
