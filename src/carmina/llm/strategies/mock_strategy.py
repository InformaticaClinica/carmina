import logging
logger = logging.getLogger(__name__)
import os
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.cloud_providers.mock_provider import MockProvider
from src.carmina.llm.utils.token_counter import get_token_counter

class MockLLMStrategy(BaseLLMStrategy):
    """
    Mock LLM strategy for testing purposes.
    This class simulates the behavior of a real LLM strategy without
    requiring an actual model or cloud provider.
    """
    __mock_identify__ = {
        "El paciente Juan García fue atendido." : 
        "El paciente [**Juan García**] fue atendido."
    }
    __mock_label__ = {
        "El paciente [**Juan García**] fue atendido." : 
        "El paciente [**OTROS_SUJETO_ASISTENCIA**] fue atendido."
    }

    def __init__(self, model_name: str="MockLLMStrategy", cloud_provider: BaseCloudProvider = None, **kwargs):
        cloud_provider = MockProvider()
        super().__init__(model_name=model_name, cloud_provider=cloud_provider, **kwargs)
        self.token_counter = get_token_counter(self.model_name, "mock")

    def get_context_window(self) -> int:
        """
        Get the maximum context window size for this model.
        
        Returns:
            Maximum number of tokens the model can process
        """
        # Mocked context window size
        return 4096
    
    def run_inference(self, text, dict_to_obtain):
        if text in dict_to_obtain:
                return dict_to_obtain[text]
        else:
            raise ValueError("Text not found in mock examples.")

    def identify(self, text: str, **kwargs) -> str:
        try:
            # Simulate the identification process
            return self.run_inference(text, self.__mock_identify__)
        except Exception as e:
            # Handle any exceptions that occur during the identification process
            print(f"Error during identification: {e}")
            return text
    
    def batch_identify(self, texts, **kwargs):
        pass

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text using mock token counter.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        return self.token_counter.count_tokens(text)
    
    def count_prompt_tokens(self, anonymization_mode: str, input_text: str) -> dict:
        """
        Count tokens for both system and user prompts.
        
        Args:
            anonymization_mode: The anonymization mode (identify, label, substitute)
            input_text: The input text to be processed
            
        Returns:
            Dictionary with token counts for system, user, and total
        """
        try:
            # Get the messages that would be sent to the model
            messages = self.get_message(anonymization_mode, input_text)
            return self.token_counter.count_message_tokens(messages)
        except Exception as e:
            logger.warning(f"Error counting prompt tokens: {e}")
            # Return approximation if message creation fails
            return {
                "system": self.count_tokens("System prompt placeholder"),
                "user": self.count_tokens(input_text),
                "total": self.count_tokens("System prompt placeholder") + self.count_tokens(input_text)
            }

    def process_for_anonymization(self, text, mode):
        try:
            # Simulate the identification process
            if mode == "label":
                return self.run_inference(text, self.__mock_label__)
            elif mode == "substitute":
                return self.run_inference(text, self.__mock_identify__)
            else:
                raise ValueError("Invalid mode. Use 'label' or 'identify'.")
        except Exception as e:
            # Handle any exceptions that occur during the identification process
            print(f"Error during identification: {e}")
            return text