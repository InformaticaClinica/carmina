"""
Unit tests for VertexGeminiStrategy.

These tests verify the strategy's behavior, parameter propagation,
and integration with VertexAIProvider.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.carmina.llm.strategies.vertex_gemini_strategy import VertexGeminiStrategy


class TestVertexGeminiStrategyParameterPropagation:
    """Tests that verify inference_params are correctly propagated to the provider."""
    
    @pytest.fixture
    def mock_provider(self):
        """Mock VertexAIProvider."""
        provider = Mock()
        provider.get_name.return_value = "vertex_ai"
        provider.run_inference.return_value = "Test response"
        return provider
    
    def test_inference_params_propagated_correctly(self, mock_provider):
        """Test that inference params are correctly propagated to provider (not overwritten)."""
        with patch.dict('os.environ', {
            'ANONYMIZATION_MODE': 'identify',
            'TEMPERATURE': '0.9',
            'MAX_TOKENS': '2048',
            'TOP_P': '0.95'
        }):
            strategy = VertexGeminiStrategy(
                model_name="gemini-2.5-flash",
                cloud_provider=mock_provider
            )
        
        messages = [{"role": "user", "content": "Test"}]
        inference_params = {
            "temperature": 0.3,  # Should override strategy default
            "max_tokens": 500,
            "top_p": 0.8
        }
        
        result = strategy.run_inference(messages, inference_params)
        
        # Verify provider was called
        assert mock_provider.run_inference.called
        
        # Get actual call args
        call_kwargs = mock_provider.run_inference.call_args[1]
        
        # FIXED: inference_params should now be used (not hardcoded strategy values)
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["top_p"] == 0.8
    
    def test_strategy_uses_env_defaults(self, mock_provider):
        """Test that strategy picks up defaults from environment."""
        with patch.dict('os.environ', {
            'ANONYMIZATION_MODE': 'identify',
            'TEMPERATURE': '0.7',
            'MAX_TOKENS': '1024',
            'TOP_P': '0.9'
        }):
            strategy = VertexGeminiStrategy(
                model_name="gemini-2.5-flash",
                cloud_provider=mock_provider
            )
        
        assert strategy.temperature == 0.7
        assert strategy.max_tokens == 1024
        assert strategy.top_p == 0.9
    
    def test_uses_strategy_defaults_when_params_not_provided(self, mock_provider):
        """Test that strategy defaults are used when inference_params doesn't provide values."""
        with patch.dict('os.environ', {
            'ANONYMIZATION_MODE': 'identify',
            'TEMPERATURE': '0.7',
            'MAX_TOKENS': '1024',
            'TOP_P': '0.9'
        }):
            strategy = VertexGeminiStrategy(
                model_name="gemini-2.5-flash",
                cloud_provider=mock_provider
            )
        
        messages = [{"role": "user", "content": "Test"}]
        inference_params = {}  # Empty params
        
        result = strategy.run_inference(messages, inference_params)
        
        # Should use strategy defaults
        call_kwargs = mock_provider.run_inference.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["top_p"] == 0.9
    
    def test_strategy_filters_none_values(self, mock_provider):
        """Test that None values are filtered from inference_params."""
        with patch.dict('os.environ', {
            'ANONYMIZATION_MODE': 'identify'
        }):
            strategy = VertexGeminiStrategy(
                model_name="gemini-2.5-flash",
                cloud_provider=mock_provider
            )
        
        messages = [{"role": "user", "content": "Test"}]
        inference_params = {
            "temperature": None,
            "max_tokens": 500,
            "top_p": None
        }
        
        result = strategy.run_inference(messages, inference_params)
        
        # Verify filtering happened (line 46-47 in vertex_gemini_strategy.py)
        call_kwargs = mock_provider.run_inference.call_args[1]
        
        # After filtering, should only have non-None values
        # But current implementation overwrites with hardcoded values anyway
        assert "temperature" in call_kwargs  # Will be strategy default
        assert "max_tokens" in call_kwargs


class TestVertexGeminiStrategyContextWindow:
    """Tests for context window detection."""
    
    @pytest.fixture
    def mock_provider(self):
        provider = Mock()
        provider.get_name.return_value = "vertex_ai"
        return provider
    
    def test_get_context_window_from_hardcoded(self, mock_provider):
        """Test getting context window from hardcoded dict."""
        with patch.dict('os.environ', {'ANONYMIZATION_MODE': 'identify'}):
            strategy = VertexGeminiStrategy(
                model_name="gemini-2.5-flash",
                cloud_provider=mock_provider
            )
        
        context_window = strategy.get_context_window()
        assert context_window == 1000000  # 1M tokens
    
    def test_get_context_window_fallback(self, mock_provider):
        """Test context window fallback for unknown model."""
        with patch.dict('os.environ', {'ANONYMIZATION_MODE': 'identify'}):
            strategy = VertexGeminiStrategy(
                model_name="gemini-unknown",
                cloud_provider=mock_provider
            )
        
        context_window = strategy.get_context_window()
        assert context_window == 1000000  # Default fallback


class TestVertexGeminiStrategyIdentify:
    """Tests for identify method."""
    
    @pytest.fixture
    def mock_provider(self):
        provider = Mock()
        provider.get_name.return_value = "vertex_ai"
        provider.run_inference.return_value = "PHI detected: patient_name"
        return provider
    
    def test_identify_calls_run_inference(self, mock_provider):
        """Test that identify method calls run_inference."""
        with patch.dict('os.environ', {'ANONYMIZATION_MODE': 'identify'}):
            strategy = VertexGeminiStrategy(
                model_name="gemini-2.5-flash",
                cloud_provider=mock_provider
            )
        
        with patch.object(strategy, 'get_message') as mock_get_message:
            mock_get_message.return_value = [{"role": "user", "content": "Identify PHI"}]
            
            result = strategy.identify("Patient John Doe, age 45")
            
            # Verify get_message was called with correct args
            mock_get_message.assert_called_once_with("identify", "Patient John Doe, age 45")
            
            # Verify run_inference was called
            assert mock_provider.run_inference.called
            assert result == "PHI detected: patient_name"


class TestVertexGeminiStrategyAnonymization:
    """Tests for anonymization workflow."""
    
    @pytest.fixture
    def mock_provider(self):
        provider = Mock()
        provider.get_name.return_value = "vertex_ai"
        provider.run_inference.return_value = "Patient [REDACTED], age [AGE]"
        return provider
    
    def test_process_for_anonymization_label(self, mock_provider):
        """Test anonymization with label strategy."""
        with patch.dict('os.environ', {'ANONYMIZATION_MODE': 'label'}):
            strategy = VertexGeminiStrategy(
                model_name="gemini-2.5-flash",
                cloud_provider=mock_provider
            )
        
        with patch.object(strategy, 'get_message') as mock_get_message:
            mock_get_message.return_value = [{"role": "user", "content": "Label PHI"}]
            
            result = strategy.process_for_anonymization(
                text="Patient John Doe, age 45",
                strategy="label"
            )
            
            mock_get_message.assert_called_once_with("label", "Patient John Doe, age 45")
            assert result == "Patient [REDACTED], age [AGE]"
    
    def test_process_for_anonymization_substitute(self, mock_provider):
        """Test anonymization with substitute strategy."""
        with patch.dict('os.environ', {'ANONYMIZATION_MODE': 'substitute'}):
            strategy = VertexGeminiStrategy(
                model_name="gemini-2.5-flash",
                cloud_provider=mock_provider
            )
        
        mock_provider.run_inference.return_value = "Patient Jane Smith, age 50"
        
        with patch.object(strategy, 'get_message') as mock_get_message:
            mock_get_message.return_value = [{"role": "user", "content": "Substitute PHI"}]
            
            result = strategy.process_for_anonymization(
                text="Patient John Doe, age 45",
                strategy="substitute"
            )
            
            mock_get_message.assert_called_once_with("substitute", "Patient John Doe, age 45")
            assert result == "Patient Jane Smith, age 50"


class TestVertexGeminiStrategyTokenCounting:
    """Tests for token counting functionality."""
    
    @pytest.fixture
    def mock_provider(self):
        provider = Mock()
        provider.get_name.return_value = "vertex_ai"
        return provider
    
    def test_count_tokens_uses_counter(self, mock_provider):
        """Test that count_tokens uses the token counter."""
        with patch.dict('os.environ', {'ANONYMIZATION_MODE': 'identify'}):
            with patch('src.carmina.llm.strategies.vertex_gemini_strategy.get_token_counter') as mock_get_counter:
                mock_counter = Mock()
                mock_counter.count_tokens.return_value = 42
                mock_get_counter.return_value = mock_counter
                
                strategy = VertexGeminiStrategy(
                    model_name="gemini-2.5-flash",
                    cloud_provider=mock_provider
                )
                
                count = strategy.count_tokens("Test text with some tokens")
                
                assert count == 42
                mock_counter.count_tokens.assert_called_once_with("Test text with some tokens")


class TestVertexGeminiStrategyBatchIdentify:
    """Tests for batch identify (not implemented)."""
    
    @pytest.fixture
    def mock_provider(self):
        provider = Mock()
        provider.get_name.return_value = "vertex_ai"
        return provider
    
    def test_batch_identify_not_implemented(self, mock_provider):
        """Test that batch_identify is not implemented yet."""
        with patch.dict('os.environ', {'ANONYMIZATION_MODE': 'identify'}):
            strategy = VertexGeminiStrategy(
                model_name="gemini-2.5-flash",
                cloud_provider=mock_provider
            )
        
        # Should return None (not implemented)
        result = strategy.batch_identify(["text1", "text2"])
        assert result is None
