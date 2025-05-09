import os
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.cloud_providers.mock_provider import MockProvider

class MockLLMStrategy(BaseLLMStrategy):
    """
    Mock LLM strategy for testing purposes.
    This class simulates the behavior of a real LLM strategy without
    requiring an actual model or cloud provider.
    """
    def __init__(self, model_name: str="MockLLMStrategy", cloud_provider: BaseCloudProvider = None, **kwargs):
        cloud_provider = MockProvider()
        super().__init__(model_name=model_name, cloud_provider=cloud_provider, **kwargs)

    def get_context_window(self) -> int:
        """
        Get the maximum context window size for this model.
        
        Returns:
            Maximum number of tokens the model can process
        """
        # Mocked context window size
        return 4096
    
    def identify(self, text: str, **kwargs) -> str:
        return text
    
    def batch_identify(self, texts, **kwargs):
        pass

    def count_tokens(self, text):
        pass

    def process_for_anonymization(self, text, mode):
        pass

    def process_for_anonymization(self, text, mode):
        pass