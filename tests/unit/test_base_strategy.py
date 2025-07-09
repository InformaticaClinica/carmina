"""
Unit tests for base LLM strategy functionality.
"""
import pytest
import os
from unittest.mock import MagicMock, patch
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy


# Create a concrete implementation for testing
class TestStrategy(BaseLLMStrategy):
    """Test implementation of BaseLLMStrategy."""
    
    def run_inference(self, messages, **kwargs):
        return "test response"
    
    def get_context_window(self):
        return 4096
    
    def count_tokens(self, text):
        return len(text.split())


@pytest.mark.unit
class TestBaseLLMStrategy:
    """Test base LLM strategy functionality."""
    
    def test_init_with_defaults(self, mock_cloud_provider):
        """Test strategy initialization with default parameters."""
        strategy = TestStrategy(
            model_name="test-model",
            cloud_provider=mock_cloud_provider
        )
        
        assert strategy.model_name == "test-model"
        assert strategy.cloud_provider == mock_cloud_provider
        assert strategy.anonymization_mode == "identify"
        assert strategy.temperature == 1.0
        assert strategy.max_tokens == 2500
        assert strategy.top_k == 40
        assert strategy.top_p == 0.95
    
    def test_init_with_custom_params(self, mock_cloud_provider):
        """Test strategy initialization with custom parameters."""
        strategy = TestStrategy(
            model_name="test-model",
            cloud_provider=mock_cloud_provider,
            anonymization_mode="label",
            temperature=0.5,
            max_tokens=1000,
            top_k=20,
            top_p=0.8
        )
        
        assert strategy.anonymization_mode == "label"
        assert strategy.temperature == 0.5
        assert strategy.max_tokens == 1000
        assert strategy.top_k == 20
        assert strategy.top_p == 0.8
    
    def test_init_with_environment_variables(self, mock_cloud_provider):
        """Test strategy initialization with environment variables."""
        with patch.dict(os.environ, {
            'ANONYMIZATION_MODE': 'substitute',
            'TEMPERATURE': '0.3',
            'MAX_TOKENS': '1500'
        }):
            strategy = TestStrategy(
                model_name="test-model",
                cloud_provider=mock_cloud_provider
            )
            
            assert strategy.anonymization_mode == "substitute"
            assert strategy.temperature == 0.3
            assert strategy.max_tokens == 1500
    
    def test_get_name(self, mock_cloud_provider):
        """Test getting strategy name."""
        strategy = TestStrategy(
            model_name="test-model",
            cloud_provider=mock_cloud_provider
        )
        
        name = strategy.get_name()
        assert "test-model" in name
        assert "TestStrategy" in name
    
    @patch('src.carmina.llm.utils.prompt_loader.load_system_prompt')
    def test_get_message_identify_mode(self, mock_load_prompt, mock_cloud_provider):
        """Test getting message for identify mode."""
        mock_load_prompt.side_effect = [
            "System prompt for identification",
            "User prompt with <input_text> placeholder"
        ]
        
        strategy = TestStrategy(
            model_name="test-model",
            cloud_provider=mock_cloud_provider,
            anonymization_mode="identify"
        )
        
        messages = strategy.get_message("identify", "Test text")
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt for identification"
        assert messages[1]["role"] == "user"
        assert "Test text" in messages[1]["content"]
    
    @patch('src.carmina.llm.utils.prompt_loader.load_system_prompt')
    def test_get_message_label_mode(self, mock_load_prompt, mock_cloud_provider):
        """Test getting message for label mode."""
        mock_load_prompt.side_effect = [
            "System prompt for labeling",
            "User prompt with <input_text> placeholder"
        ]
        
        strategy = TestStrategy(
            model_name="test-model",
            cloud_provider=mock_cloud_provider,
            anonymization_mode="label"
        )
        
        messages = strategy.get_message("label", "Test text")
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt for labeling"
        assert messages[1]["role"] == "user"
        assert "Test text" in messages[1]["content"]
    
    def test_identify(self, mock_cloud_provider):
        """Test identify method."""
        strategy = TestStrategy(
            model_name="test-model",
            cloud_provider=mock_cloud_provider
        )
        
        with patch.object(strategy, 'get_message') as mock_get_message, \
             patch.object(strategy, 'run_inference') as mock_run_inference:
            
            mock_get_message.return_value = [{"role": "user", "content": "test"}]
            mock_run_inference.return_value = "identified text"
            
            result = strategy.identify("Test text")
            
            mock_get_message.assert_called_once_with("identify", "Test text")
            mock_run_inference.assert_called_once()
            assert result == "identified text"
    
    def test_label(self, mock_cloud_provider):
        """Test label method."""
        strategy = TestStrategy(
            model_name="test-model",
            cloud_provider=mock_cloud_provider
        )
        
        with patch.object(strategy, 'get_message') as mock_get_message, \
             patch.object(strategy, 'run_inference') as mock_run_inference:
            
            mock_get_message.return_value = [{"role": "user", "content": "test"}]
            mock_run_inference.return_value = "labeled text"
            
            result = strategy.label("Test text")
            
            mock_get_message.assert_called_once_with("label", "Test text")
            mock_run_inference.assert_called_once()
            assert result == "labeled text"
    
    def test_substitute(self, mock_cloud_provider):
        """Test substitute method."""
        strategy = TestStrategy(
            model_name="test-model",
            cloud_provider=mock_cloud_provider
        )
        
        with patch.object(strategy, 'get_message') as mock_get_message, \
             patch.object(strategy, 'run_inference') as mock_run_inference:
            
            mock_get_message.return_value = [{"role": "user", "content": "test"}]
            mock_run_inference.return_value = "substituted text"
            
            result = strategy.substitute("Test text")
            
            mock_get_message.assert_called_once_with("substitute", "Test text")
            mock_run_inference.assert_called_once()
            assert result == "substituted text"
    
    def test_process_for_anonymization_identify(self, mock_cloud_provider):
        """Test process_for_anonymization in identify mode."""
        strategy = TestStrategy(
            model_name="test-model",
            cloud_provider=mock_cloud_provider,
            anonymization_mode="identify"
        )
        
        with patch.object(strategy, 'identify') as mock_identify:
            mock_identify.return_value = "identified text"
            
            result = strategy.process_for_anonymization("Test text")
            
            mock_identify.assert_called_once_with("Test text")
            assert result == "identified text"
    
    def test_process_for_anonymization_label(self, mock_cloud_provider):
        """Test process_for_anonymization in label mode."""
        strategy = TestStrategy(
            model_name="test-model",
            cloud_provider=mock_cloud_provider,
            anonymization_mode="label"
        )
        
        with patch.object(strategy, 'label') as mock_label:
            mock_label.return_value = "labeled text"
            
            result = strategy.process_for_anonymization("Test text")
            
            mock_label.assert_called_once_with("Test text")
            assert result == "labeled text"
    
    def test_process_for_anonymization_substitute(self, mock_cloud_provider):
        """Test process_for_anonymization in substitute mode."""
        strategy = TestStrategy(
            model_name="test-model",
            cloud_provider=mock_cloud_provider,
            anonymization_mode="substitute"
        )
        
        with patch.object(strategy, 'substitute') as mock_substitute:
            mock_substitute.return_value = "substituted text"
            
            result = strategy.process_for_anonymization("Test text")
            
            mock_substitute.assert_called_once_with("Test text")
            assert result == "substituted text"
    
    def test_process_for_anonymization_invalid_mode(self, mock_cloud_provider):
        """Test process_for_anonymization with invalid mode."""
        strategy = TestStrategy(
            model_name="test-model",
            cloud_provider=mock_cloud_provider,
            anonymization_mode="invalid"
        )
        
        with pytest.raises(ValueError, match="Unknown anonymization mode"):
            strategy.process_for_anonymization("Test text")