"""
Unit tests for enhanced Vertex AI Provider features.

Tests for:
- Retry with exponential backoff
- External model ID mapping configuration
- Fallback strategy for unmapped models
- Safety block detection
- Timeout configuration
"""

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable
from src.carmina.llm.cloud_providers.vertex_ai_provider import VertexAIProvider, SafetyBlockError


class TestVertexAIProviderRetry:
    """Tests for retry mechanism with exponential backoff."""
    
    @pytest.fixture
    def mock_genai_client(self):
        with patch('src.carmina.llm.cloud_providers.vertex_ai_provider.genai.Client') as mock_client:
            yield mock_client
    
    @pytest.fixture
    def mock_default_credentials(self):
        with patch('src.carmina.llm.cloud_providers.vertex_ai_provider.default') as mock_default:
            mock_credentials = Mock()
            mock_default.return_value = (mock_credentials, "test-project")
            yield mock_default
    
    @pytest.fixture
    def valid_env_vars(self):
        env_vars = {
            "VERTEX_PROJECT_ID": "test-project-123",
            "VERTEX_LOCATION": "europe-west4"
        }
        with patch.dict(os.environ, env_vars, clear=False):
            yield env_vars
    
    def test_retry_on_resource_exhausted(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test that retry mechanism works for ResourceExhausted (429) errors."""
        mock_client_instance = mock_genai_client.return_value
        
        # First two calls fail with 429, third succeeds
        mock_response = Mock()
        mock_response.text = "Success after retry"
        mock_response.candidates = []  # Empty candidates to skip safety check
        mock_client_instance.models.generate_content.side_effect = [
            ResourceExhausted("Quota exceeded"),
            ResourceExhausted("Quota exceeded"),
            mock_response
        ]
        
        provider = VertexAIProvider()
        messages = [{"role": "user", "content": "Test"}]
        
        # Should retry and eventually succeed
        result = provider.run_inference(
            model_id="gemini-2.0-flash",
            messages=messages
        )
        
        assert result == "Success after retry"
        # Should have been called 3 times (2 retries + 1 success)
        assert mock_client_instance.models.generate_content.call_count == 3
    
    def test_retry_on_internal_server_error(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test that retry mechanism works for InternalServerError (5xx) errors."""
        mock_client_instance = mock_genai_client.return_value
        
        # First call fails with 500, second succeeds
        mock_response = Mock()
        mock_response.text = "Success after server recovery"
        mock_response.candidates = []  # Empty candidates to skip safety check
        mock_client_instance.models.generate_content.side_effect = [
            InternalServerError("Server error"),
            mock_response
        ]
        
        provider = VertexAIProvider()
        messages = [{"role": "user", "content": "Test"}]
        
        result = provider.run_inference(
            model_id="gemini-2.0-flash",
            messages=messages
        )
        
        assert result == "Success after server recovery"
        assert mock_client_instance.models.generate_content.call_count == 2
    
    def test_retry_exhausted_raises_error(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test that error is raised after max retries."""
        mock_client_instance = mock_genai_client.return_value
        
        # All attempts fail
        mock_client_instance.models.generate_content.side_effect = ResourceExhausted("Quota exceeded")
        
        provider = VertexAIProvider()
        messages = [{"role": "user", "content": "Test"}]
        
        # Should raise after 3 attempts
        with pytest.raises(ResourceExhausted, match="Quota exceeded"):
            provider.run_inference(
                model_id="gemini-2.0-flash",
                messages=messages
            )
        
        # Should have tried 3 times
        assert mock_client_instance.models.generate_content.call_count == 3
    
    def test_no_retry_on_value_error(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test that ValueError does not trigger retry."""
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.side_effect = ValueError("Invalid input")
        
        provider = VertexAIProvider()
        messages = [{"role": "user", "content": "Test"}]
        
        # Should fail immediately without retry
        with pytest.raises(ValueError, match="Invalid input"):
            provider.run_inference(
                model_id="gemini-2.0-flash",
                messages=messages
            )
        
        # Should have been called only once (no retry)
        assert mock_client_instance.models.generate_content.call_count == 1


class TestVertexAIProviderModelMappings:
    """Tests for external model ID mapping configuration."""
    
    @pytest.fixture
    def mock_genai_client(self):
        with patch('src.carmina.llm.cloud_providers.vertex_ai_provider.genai.Client') as mock_client:
            yield mock_client
    
    @pytest.fixture
    def mock_default_credentials(self):
        with patch('src.carmina.llm.cloud_providers.vertex_ai_provider.default') as mock_default:
            mock_credentials = Mock()
            mock_default.return_value = (mock_credentials, "test-project")
            yield mock_default
    
    @pytest.fixture
    def valid_env_vars(self):
        env_vars = {
            "VERTEX_PROJECT_ID": "test-project-123",
            "VERTEX_LOCATION": "europe-west4"
        }
        with patch.dict(os.environ, env_vars, clear=False):
            yield env_vars
    
    def test_load_mappings_from_config_file(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test loading model mappings from external JSON config."""
        config_content = json.dumps({
            "mappings": {
                "gemini-test": "gemini-test-v1",
                "gemini-custom": "gemini-custom-v2"
            }
        })
        
        # Reset class variable to force reload
        VertexAIProvider._config_loaded = False
        
        with patch('builtins.open', mock_open(read_data=config_content)):
            with patch('pathlib.Path.exists', return_value=True):
                provider = VertexAIProvider()
                
                assert VertexAIProvider._vertex_model_ids["gemini-test"] == "gemini-test-v1"
                assert VertexAIProvider._vertex_model_ids["gemini-custom"] == "gemini-custom-v2"
    
    def test_fallback_to_defaults_when_config_missing(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test fallback to default mappings when config file is missing."""
        VertexAIProvider._config_loaded = False
        
        with patch('pathlib.Path.exists', return_value=False):
            provider = VertexAIProvider()
            
            # Should use defaults
            assert "gemini-2.0-flash" in VertexAIProvider._vertex_model_ids
            assert VertexAIProvider._vertex_model_ids["gemini-2.0-flash"] == "gemini-2.0-flash"
    
    def test_unmapped_model_uses_fallback(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test that unmapped models use fallback strategy (use ID as-is)."""
        mock_response = Mock()
        mock_response.text = "Response from unmapped model"
        mock_response.candidates = []  # Empty candidates to skip safety check
        
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.return_value = mock_response
        
        provider = VertexAIProvider()
        messages = [{"role": "user", "content": "Test"}]
        
        # Use an unmapped model ID
        result = provider.run_inference(
            model_id="gemini-experimental-new",
            messages=messages
        )
        
        assert result == "Response from unmapped model"
        
        # Verify the unmapped ID was used as-is
        call_args = mock_client_instance.models.generate_content.call_args
        assert call_args[1]["model"] == "gemini-experimental-new"


class TestVertexAIProviderSafetyBlocks:
    """Tests for safety block detection."""
    
    @pytest.fixture
    def mock_genai_client(self):
        with patch('src.carmina.llm.cloud_providers.vertex_ai_provider.genai.Client') as mock_client:
            yield mock_client
    
    @pytest.fixture
    def mock_default_credentials(self):
        with patch('src.carmina.llm.cloud_providers.vertex_ai_provider.default') as mock_default:
            mock_credentials = Mock()
            mock_default.return_value = (mock_credentials, "test-project")
            yield mock_default
    
    @pytest.fixture
    def valid_env_vars(self):
        env_vars = {
            "VERTEX_PROJECT_ID": "test-project-123",
            "VERTEX_LOCATION": "europe-west4"
        }
        with patch.dict(os.environ, env_vars, clear=False):
            yield env_vars
    
    def test_safety_block_raises_explicit_error(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test that safety blocks are detected and raise SafetyBlockError."""
        mock_candidate = Mock()
        mock_candidate.finish_reason = "SAFETY"
        
        mock_response = Mock()
        mock_response.text = None
        mock_response.candidates = [mock_candidate]
        
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.return_value = mock_response
        
        provider = VertexAIProvider()
        messages = [{"role": "user", "content": "Dangerous content"}]
        
        with pytest.raises(SafetyBlockError, match="Response blocked by safety filters"):
            provider.run_inference(
                model_id="gemini-2.0-flash",
                messages=messages
            )
    
    def test_empty_response_without_safety_block(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test handling of empty response without safety block."""
        mock_response = Mock()
        mock_response.text = ""
        mock_response.candidates = []
        
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.return_value = mock_response
        
        provider = VertexAIProvider()
        messages = [{"role": "user", "content": "Test"}]
        
        # Empty string is valid (should not raise)
        result = provider.run_inference(
            model_id="gemini-2.0-flash",
            messages=messages
        )
        
        assert result == ""
    
    def test_extract_text_from_candidates_when_text_missing(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test extracting text from candidates when .text attribute is missing."""
        mock_part = Mock()
        mock_part.text = "Text from candidate part"
        
        mock_content = Mock()
        mock_content.parts = [mock_part]
        
        mock_candidate = Mock()
        mock_candidate.content = mock_content
        
        mock_response = Mock(spec=[])  # No .text attribute
        mock_response.candidates = [mock_candidate]
        
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.return_value = mock_response
        
        provider = VertexAIProvider()
        messages = [{"role": "user", "content": "Test"}]
        
        result = provider.run_inference(
            model_id="gemini-2.0-flash",
            messages=messages
        )
        
        assert result == "Text from candidate part"


class TestVertexAIProviderTimeout:
    """Tests for timeout configuration."""
    
    @pytest.fixture
    def mock_genai_client(self):
        with patch('src.carmina.llm.cloud_providers.vertex_ai_provider.genai.Client') as mock_client:
            yield mock_client
    
    @pytest.fixture
    def mock_default_credentials(self):
        with patch('src.carmina.llm.cloud_providers.vertex_ai_provider.default') as mock_default:
            mock_credentials = Mock()
            mock_default.return_value = (mock_credentials, "test-project")
            yield mock_default
    
    def test_default_timeout_is_60_seconds(self, mock_genai_client, mock_default_credentials):
        """Test that default timeout is 60 seconds."""
        with patch.dict(os.environ, {"VERTEX_PROJECT_ID": "test-project"}):
            provider = VertexAIProvider()
            assert provider.timeout == 60
    
    def test_custom_timeout_from_kwargs(self, mock_genai_client, mock_default_credentials):
        """Test setting custom timeout via kwargs."""
        with patch.dict(os.environ, {"VERTEX_PROJECT_ID": "test-project"}):
            provider = VertexAIProvider(timeout=120)
            assert provider.timeout == 120
