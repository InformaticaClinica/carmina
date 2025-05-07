import os

class MockLLMStrategy:
    def __init__(self, model_name: str, cloud_provider: str, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)
        self.provider_name = self.cloud_provider.get_name()
        self.anonymization_mode =  os.environ.get("ANONYMIZATION_MODE") or kwargs.get("anonymization_mode")
        self.temperature = os.environ.get("TEMPERATURE") or kwargs.get("temperature")
        self.max_tokens = os.environ.get("MAX_TOKENS") or kwargs.get("max_tokens")
        self.top_p = os.environ.get("TOP_P") or kwargs.get("top_p")
        self.frequency_penalty = os.environ.get("FREQUENCY_PENALTY") or kwargs.get("frequency_penalty")
        self.presence_penalty = os.environ.get("PRESENCE_PENALTY") or kwargs.get("presence_penalty")

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