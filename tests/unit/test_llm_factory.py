"""
Unit tests for LLM factory functionality.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.carmina.llm.factory import LLMFactory
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy


@pytest.mark.unit
class TestLLMFactory:
    """Test LLM factory functionality."""
    
    def test_get_strategy_for_model_openai(self):
        """Test getting strategy for OpenAI models."""
        strategy_class = LLMFactory._get_strategy_for_model("gpt-4-turbo")
        assert strategy_class.__name__ == "OpenAIStrategy"
    
    def test_get_strategy_for_model_anthropic(self):
        """Test getting strategy for Anthropic models."""
        strategy_class = LLMFactory._get_strategy_for_model("claude-3.5-sonnet")
        assert strategy_class.__name__ == "AnthropicStrategy"
    
    def test_get_strategy_for_model_gemini(self):
        """Test getting strategy for Gemini models."""
        strategy_class = LLMFactory._get_strategy_for_model("gemini-2.0-flash")
        assert strategy_class.__name__ == "GeminiStrategy"
    
    def test_get_strategy_for_model_deepseek(self):
        """Test getting strategy for DeepSeek models."""
        strategy_class = LLMFactory._get_strategy_for_model("deepseek-v3")
        assert strategy_class.__name__ == "DeepseekStrategy"
    
    def test_get_strategy_for_model_llama(self):
        """Test getting strategy for Llama models."""
        strategy_class = LLMFactory._get_strategy_for_model("llama-3.3-70b")
        assert strategy_class.__name__ == "LlamaStrategy"
    
    def test_get_strategy_for_model_unknown(self):
        """Test getting strategy for unknown model raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            LLMFactory._get_strategy_for_model("unknown-model")
    
    @patch('src.carmina.llm.factory.CloudProviderFactory')
    def test_create_llm_strategy(self, mock_provider_factory):
        """Test creating LLM strategy."""
        # Mock cloud provider
        mock_provider = MagicMock()
        mock_provider_factory.create.return_value = mock_provider
        
        # Mock strategy class
        mock_strategy = MagicMock()
        
        with patch.object(LLMFactory, '_get_strategy_for_model') as mock_get_strategy:
            mock_strategy_class = MagicMock()
            mock_strategy_class.return_value = mock_strategy
            mock_get_strategy.return_value = mock_strategy_class
            
            result = LLMFactory.create(
                model_name="test-model",
                cloud_provider="test-provider",
                strategy_kwargs={"anonymization_mode": "identify"},
                provider_kwargs={"api_key": "test-key"}
            )
            
            # Verify provider factory was called
            mock_provider_factory.create.assert_called_once_with(
                "test-provider", api_key="test-key"
            )
            
            # Verify strategy was created
            mock_strategy_class.assert_called_once_with(
                model_name="test-model",
                cloud_provider=mock_provider,
                anonymization_mode="identify"
            )
            
            assert result == mock_strategy
    
    def test_create_with_defaults(self):
        """Test creating LLM strategy with default parameters."""
        with patch('src.carmina.llm.factory.CloudProviderFactory') as mock_provider_factory, \
             patch.object(LLMFactory, '_get_strategy_for_model') as mock_get_strategy:
            
            mock_provider = MagicMock()
            mock_provider_factory.create.return_value = mock_provider
            
            mock_strategy = MagicMock()
            mock_strategy_class = MagicMock()
            mock_strategy_class.return_value = mock_strategy
            mock_get_strategy.return_value = mock_strategy_class
            
            result = LLMFactory.create(
                model_name="claude-3.5-sonnet",
                cloud_provider="aws"
            )
            
            # Verify provider factory was called with empty kwargs
            mock_provider_factory.create.assert_called_once_with("aws")
            
            # Verify strategy was created with empty kwargs
            mock_strategy_class.assert_called_once_with(
                model_name="claude-3.5-sonnet",
                cloud_provider=mock_provider
            )
            
            assert result == mock_strategy