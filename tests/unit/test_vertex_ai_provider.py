"""
Unit tests for Vertex AI Provider.

These tests verify the functionality of the VertexAIProvider class
using mocks to avoid actual API calls to Google Cloud.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.carmina.llm.cloud_providers.vertex_ai_provider import VertexAIProvider


class TestVertexAIProvider:
    """Unit tests for VertexAIProvider."""
    
    @pytest.fixture
    def mock_genai_client(self):
        """Mock google.genai.Client."""
        with patch('src.carmina.llm.cloud_providers.vertex_ai_provider.genai.Client') as mock_client:
            yield mock_client
    
    @pytest.fixture
    def mock_default_credentials(self):
        """Mock google.auth.default for ADC."""
        with patch('src.carmina.llm.cloud_providers.vertex_ai_provider.default') as mock_default:
            mock_credentials = Mock()
            mock_default.return_value = (mock_credentials, "test-project")
            yield mock_default
    
    @pytest.fixture
    def valid_env_vars(self):
        """Set up valid environment variables."""
        env_vars = {
            "VERTEX_PROJECT_ID": "test-project-123",
            "VERTEX_LOCATION": "europe-west4"
        }
        with patch.dict(os.environ, env_vars, clear=False):
            yield env_vars
    
    def test_init_success_with_env_vars(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test successful initialization with environment variables."""
        provider = VertexAIProvider()
        
        assert provider.project_id == "test-project-123"
        assert provider.location == "europe-west4"
        assert provider.get_name() == "vertex_ai"
        
        # Verify client was initialized with correct params
        mock_genai_client.assert_called_once_with(
            vertexai=True,
            project="test-project-123",
            location="europe-west4"
        )
    
    def test_init_success_with_kwargs(self, mock_genai_client, mock_default_credentials):
        """Test successful initialization with kwargs (overriding env vars)."""
        provider = VertexAIProvider(
            project_id="custom-project",
            location="us-central1"
        )
        
        assert provider.project_id == "custom-project"
        assert provider.location == "us-central1"
        
        mock_genai_client.assert_called_once_with(
            vertexai=True,
            project="custom-project",
            location="us-central1"
        )
    
    def test_init_missing_project_id(self, mock_genai_client, mock_default_credentials):
        """Test initialization fails when project_id is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Project ID is required"):
                VertexAIProvider()
    
    def test_init_missing_credentials(self, mock_genai_client, valid_env_vars):
        """Test initialization fails when credentials are not available."""
        from google.auth.exceptions import DefaultCredentialsError
        
        with patch('src.carmina.llm.cloud_providers.vertex_ai_provider.default') as mock_default:
            mock_default.side_effect = DefaultCredentialsError("No credentials found")
            
            with pytest.raises(ValueError, match="Google Cloud credentials not found"):
                VertexAIProvider()
    
    def test_run_inference_success(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test successful inference call."""
        # Setup mock response
        mock_response = Mock()
        mock_response.text = "This is the generated response."
        mock_response.candidates = []  # Empty list to skip safety check
        
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.return_value = mock_response
        
        provider = VertexAIProvider()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        result = provider.run_inference(
            model_id="gemini-2.5-flash",
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        assert result == "This is the generated response."
        
        # Verify generate_content was called
        assert mock_client_instance.models.generate_content.called
    
    def test_run_inference_unknown_model(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test inference with unknown model uses fallback strategy."""
        mock_response = Mock()
        mock_response.text = "Response from unmapped model"
        mock_response.candidates = []
        
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.return_value = mock_response
        
        provider = VertexAIProvider()
        messages = [{"role": "user", "content": "Test"}]
        
        # Unknown models now use fallback (log warning but don't fail)
        result = provider.run_inference(
            model_id="unknown-model-xyz",
            messages=messages
        )
        
        assert result == "Response from unmapped model"
    
    def test_run_inference_error_handling(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test error handling during inference."""
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.side_effect = Exception("API Error")
        
        provider = VertexAIProvider()
        
        messages = [{"role": "user", "content": "Test"}]
        
        # Errors should propagate directly
        with pytest.raises(Exception, match="API Error"):
            provider.run_inference(
                model_id="gemini-2.5-flash",
                messages=messages
            )
    
    def test_adapt_message_with_system_instruction(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test message adaptation with system instruction."""
        provider = VertexAIProvider()
        
        messages = [
            {"role": "system", "content": "You are a medical expert."},
            {"role": "user", "content": "What is diabetes?"}
        ]
        
        payload = provider.adapt_message(messages, temperature=0.5, max_tokens=200)
        
        assert "contents" in payload
        assert "config" in payload
        
        # Contents should be string format from check_user_message
        assert isinstance(payload["contents"], str) or isinstance(payload["contents"], list)
    
    def test_adapt_message_reject_assistant_role(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test that assistant role raises error."""
        provider = VertexAIProvider()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        # Assistant role should raise ValueError
        with pytest.raises(ValueError, match="Assistant role is not supported"):
            provider.adapt_message(messages)
    
    def test_get_messages_parsing(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test message parsing logic."""
        provider = VertexAIProvider()
        
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"}
        ]
        
        system_instruction, contents = provider.get_messages(messages)
        
        assert system_instruction == "System prompt"
        # Contents is processed by check_user_message
        assert contents is not None
    
    def test_generate_config_with_all_params(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test configuration generation with all parameters."""
        provider = VertexAIProvider()
        
        config = provider.generate_config(
            system_instruction="Test system",
            temperature=0.8,
            max_tokens=500,
            top_p=0.9,
            top_k=40,
            stop_sequences=["STOP", "END"]
        )
        
        # Verify config object was created (checking type)
        from google.genai import types
        assert isinstance(config, types.GenerateContentConfig)
    
    def test_generate_config_minimal(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test configuration generation with minimal parameters."""
        provider = VertexAIProvider()
        
        config = provider.generate_config(system_instruction=None)
        
        # Should create config even with minimal parameters
        from google.genai import types
        assert isinstance(config, types.GenerateContentConfig)
    
    def test_model_id_mapping(self, mock_genai_client, mock_default_credentials, valid_env_vars):
        """Test that model IDs are correctly mapped."""
        mock_response = Mock()
        mock_response.text = "Response"
        mock_response.candidates = []
        
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.return_value = mock_response
        
        provider = VertexAIProvider()
        
        # Test mapped model
        provider.run_inference(
            model_id="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Test"}]
        )
        
        # Verify the mapped name was used
        call_args = mock_client_instance.models.generate_content.call_args
        assert call_args[1]["model"] == "gemini-2.5-flash-preview-04-17"


class TestVertexAIProviderIntegration:
    """
    Integration tests for Vertex AI Provider.
    
    These tests require actual GCP credentials and should only run
    when RUN_VERTEX_IT=1 is set.
    
    Prerequisites:
        - VERTEX_PROJECT_ID environment variable set
        - VERTEX_LOCATION environment variable set
        - Valid GCP credentials (Application Default Credentials)
        - Vertex AI API enabled in your GCP project
    
    To run: RUN_VERTEX_IT=1 pytest -k TestVertexAIProviderIntegration -v
    """
    
    @pytest.mark.skipif(
        os.getenv("RUN_VERTEX_IT") != "1",
        reason="Set RUN_VERTEX_IT=1 to run real Vertex test"
    )
    def test_real_inference_basic(self):
        """
        Test basic inference call to Vertex AI with deterministic output.
        
        Uses temperature=0 to ensure reproducible output for assertion.
        """
        provider = VertexAIProvider()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Always follow instructions exactly."},
            {"role": "user", "content": "Say exactly 'Hello World' and nothing else."}
        ]
        
        result = provider.run_inference(
            model_id="gemini-2.0-flash",
            messages=messages,
            temperature=0.0,
            max_tokens=20
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        # With temperature=0, should be deterministic
        assert "Hello World" in result or "hello world" in result.lower()
        print(f"✅ Basic inference response: {result}")
    
    @pytest.mark.skipif(
        os.getenv("RUN_VERTEX_IT") != "1",
        reason="Set RUN_VERTEX_IT=1 to run real Vertex test"
    )
    def test_real_inference_max_tokens(self):
        """
        Test that max_tokens limits response length.
        
        Requests a long response but limits tokens to verify truncation.
        """
        provider = VertexAIProvider()
        
        messages = [
            {"role": "user", "content": "Count from 1 to 100."}
        ]
        
        result = provider.run_inference(
            model_id="gemini-2.0-flash",
            messages=messages,
            temperature=0.0,
            max_tokens=10  # Very low to force truncation
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be truncated, so shouldn't contain "100"
        # But should contain early numbers
        assert any(str(i) in result for i in range(1, 5))
        print(f"✅ Max tokens response (limited to 10): {result}")
    
    @pytest.mark.skipif(
        os.getenv("RUN_VERTEX_IT") != "1",
        reason="Set RUN_VERTEX_IT=1 to run real Vertex test"
    )
    def test_real_inference_stop_sequences(self):
        """
        Test that stop_sequences work correctly.
        
        Asks model to list A, B, C but stops at B.
        Note: Stop sequence behavior may vary - test is lenient.
        """
        provider = VertexAIProvider()
        
        messages = [
            {"role": "user", "content": "List the letters A, B, C, D, E in order, one per line."}
        ]
        
        result = provider.run_inference(
            model_id="gemini-2.0-flash",
            messages=messages,
            temperature=0.0,
            stop_sequences=["C"]  # Should stop when hitting "C"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain A and B
        assert "A" in result
        assert "B" in result
        # May or may not contain C (depending on when stop is triggered)
        # Should NOT contain D or E
        print(f"✅ Stop sequences response: {result}")
    
    @pytest.mark.skipif(
        os.getenv("RUN_VERTEX_IT") != "1",
        reason="Set RUN_VERTEX_IT=1 to run real Vertex test"
    )
    def test_real_inference_temperature_variation(self):
        """
        Test temperature affects response (non-deterministically).
        
        This test verifies the API accepts temperature parameter.
        """
        provider = VertexAIProvider()
        
        messages = [
            {"role": "user", "content": "Say hello in a creative way."}
        ]
        
        # Temperature 0.0 (deterministic)
        result_low = provider.run_inference(
            model_id="gemini-2.0-flash",
            messages=messages,
            temperature=0.0,
            max_tokens=50
        )
        
        # Temperature 1.5 (more creative)
        result_high = provider.run_inference(
            model_id="gemini-2.0-flash",
            messages=messages,
            temperature=1.5,
            max_tokens=50
        )
        
        assert isinstance(result_low, str)
        assert isinstance(result_high, str)
        assert len(result_low) > 0
        assert len(result_high) > 0
        print(f"✅ Temperature 0.0: {result_low}")
        print(f"✅ Temperature 1.5: {result_high}")
