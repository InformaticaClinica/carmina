"""
Integration test for Vertex AI provider with GeminiStrategy.

This test verifies the complete integration of VertexAIProvider
with the LLMFactory and GeminiStrategy.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from src.carmina.llm.factory import LLMFactory
from src.carmina.llm.cloud_providers.cloud_provider_factory import CloudProviderFactory
from src.carmina.llm.strategies.gemini_strategy import GeminiStrategy


class TestVertexAIIntegration:
    """Integration tests for Vertex AI with LLM Factory."""
    
    @pytest.fixture
    def mock_genai_client(self):
        """Mock google.genai.Client for all tests."""
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
    def vertex_env_vars(self):
        """Set up Vertex AI environment variables."""
        env_vars = {
            "VERTEX_PROJECT_ID": "test-project-vertex",
            "VERTEX_LOCATION": "us-central1"
        }
        with patch.dict(os.environ, env_vars, clear=False):
            yield env_vars
    
    def test_cloud_provider_factory_vertex_ai(self, mock_genai_client, mock_default_credentials, vertex_env_vars):
        """Test that CloudProviderFactory can create VertexAIProvider."""
        provider = CloudProviderFactory.create("vertex_ai")
        
        assert provider.get_name() == "vertex_ai"
        assert provider.project_id == "test-project-vertex"
        assert provider.location == "us-central1"
    
    def test_llm_factory_with_vertex_ai(self, mock_genai_client, mock_default_credentials, vertex_env_vars):
        """Test LLMFactory.create with vertex_ai provider."""
        llm = LLMFactory.create(
            model_name="gemini-2.5-flash",
            cloud_provider="vertex_ai",
            strategy_kwargs={"anonymization_mode": "identify"},
            provider_kwargs={}
        )
        
        assert isinstance(llm, GeminiStrategy)
        assert llm.model_name == "gemini-2.5-flash"
        assert llm.provider_name == "vertex_ai"
        assert llm.anonymization_mode == "identify"
    
    def test_gemini_strategy_with_vertex_ai_run_inference(
        self,
        mock_genai_client,
        mock_default_credentials,
        vertex_env_vars
    ):
        """Test that GeminiStrategy can run inference with VertexAIProvider."""
        # Setup mock response
        mock_response = Mock()
        mock_response.text = "Identified entities: [**Juan García**]"
        
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.return_value = mock_response
        
        # Create LLM via factory
        llm = LLMFactory.create(
            model_name="gemini-2.5-flash",
            cloud_provider="vertex_ai",
            strategy_kwargs={
                "anonymization_mode": "identify",
                "temperature": 0.0,
                "max_tokens": 100
            },
            provider_kwargs={}
        )
        
        # Run inference
        messages = [
            {"role": "system", "content": "You are a data anonymizer."},
            {"role": "user", "content": "Identify PHI in: Juan García, DNI 12345678A"}
        ]
        
        inference_params = {
            "temperature": 0.0,
            "max_tokens": 100,
            "top_p": 1.0
        }
        
        result = llm.run_inference(messages, inference_params)
        
        assert result == "Identified entities: [**Juan García**]"
        
        # Verify generate_content was called on the mock client
        assert mock_client_instance.models.generate_content.called
        
        # Verify the call arguments
        call_kwargs = mock_client_instance.models.generate_content.call_args[1]
        assert "model" in call_kwargs
        assert "contents" in call_kwargs
        assert "config" in call_kwargs
    
    def test_gemini_strategy_identify_method(
        self,
        mock_genai_client,
        mock_default_credentials,
        vertex_env_vars
    ):
        """Test GeminiStrategy.identify() method with Vertex AI."""
        # Setup mock response
        mock_response = Mock()
        mock_response.text = "Patient [**name**] was admitted."
        
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.return_value = mock_response
        
        # Create LLM with identify mode
        llm = LLMFactory.create(
            model_name="gemini-2.0-flash",
            cloud_provider="vertex_ai",
            strategy_kwargs={"anonymization_mode": "identify"}
        )
        
        # Mock the prompt loading (to avoid file system dependency)
        with patch.object(llm, 'get_message') as mock_get_message:
            mock_get_message.return_value = [
                {"role": "system", "content": "Identify PHI"},
                {"role": "user", "content": "Patient Juan García was admitted."}
            ]
            
            result = llm.identify("Patient Juan García was admitted.")
            
            assert result == "Patient [**name**] was admitted."
            mock_get_message.assert_called_once()
    
    def test_vertex_ai_with_different_models(
        self,
        mock_genai_client,
        mock_default_credentials,
        vertex_env_vars
    ):
        """Test that different Gemini models work with Vertex AI."""
        mock_response = Mock()
        mock_response.text = "Response"
        
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.return_value = mock_response
        
        models_to_test = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro"
        ]
        
        for model_name in models_to_test:
            llm = LLMFactory.create(
                model_name=model_name,
                cloud_provider="vertex_ai"
            )
            
            assert llm.model_name == model_name
            assert llm.provider_name == "vertex_ai"
    
    def test_vertex_ai_error_propagation(
        self,
        mock_genai_client,
        mock_default_credentials,
        vertex_env_vars
    ):
        """Test that errors from Vertex AI are properly propagated."""
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.side_effect = Exception("Vertex API Error: Quota exceeded")
        
        llm = LLMFactory.create(
            model_name="gemini-2.5-flash",
            cloud_provider="vertex_ai"
        )
        
        messages = [{"role": "user", "content": "Test"}]
        inference_params = {"temperature": 0.0}
        
        with pytest.raises(Exception, match="Vertex AI inference error"):
            llm.run_inference(messages, inference_params)
    
    def test_vertex_ai_with_custom_parameters(
        self,
        mock_genai_client,
        mock_default_credentials,
        vertex_env_vars
    ):
        """Test Vertex AI with custom inference parameters."""
        mock_response = Mock()
        mock_response.text = "Custom response"
        
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content.return_value = mock_response
        
        llm = LLMFactory.create(
            model_name="gemini-2.5-pro",
            cloud_provider="vertex_ai",
            strategy_kwargs={
                "temperature": 0.8,
                "max_tokens": 2000,
                "top_p": 0.95
            }
        )
        
        assert llm.temperature == 0.8
        assert llm.max_tokens == 2000
        assert llm.top_p == 0.95
        
        messages = [{"role": "user", "content": "Test"}]
        result = llm.run_inference(messages, llm.get_inference_params())
        
        assert result == "Custom response"
