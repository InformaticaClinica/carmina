import pytest
import os
from unittest.mock import MagicMock, patch
from src.carmina.llm.cloud_providers.mock_provider import MockProvider
from src.carmina.llm.cloud_providers.local_provider import LocalProvider
from src.carmina.llm.strategies.llama_strategy import LlamaStrategy

@pytest.mark.llm
class TestLlama3_2_3bStrategy:
    @pytest.fixture
    def mock_llama_strategy(self):
        """Fixture para una estrategia LLaMA simulada"""
        mock_cloud_provider = MockProvider()
        strategy = LlamaStrategy(model_name="llama-3.2-3b", cloud_provider=mock_cloud_provider)
        return strategy
    
    @pytest.fixture
    def llama_strategy(self):
        """Fixture para la estrategia de LLaMA"""
        local_cloud_provider = LocalProvider("local", base_url="http://localhost:11434")
        return LlamaStrategy(model_name="llama-3.2-3b", cloud_provider=local_cloud_provider)

    def test_init(self, mock_llama_strategy):
        """Test que verifica la inicialización de la estrategia LLaMA"""
        assert isinstance(mock_llama_strategy.cloud_provider, MockProvider)
        assert mock_llama_strategy.anonymization_mode == "identify"
        assert mock_llama_strategy.get_name() == "llama-3.2-3b"
    
    def test_init_with_local_provider(self, llama_strategy):
        """Test que verifica la inicialización de la estrategia LLaMA con un proveedor local"""
        assert llama_strategy.anonymization_mode == "identify"
        assert llama_strategy.get_name() == "llama-3.2-3b"
    
    def test_local_conection(self, llama_strategy):
        """Test que verifica la conexión de la estrategia LLaMA"""
        assert llama_strategy.cloud_provider.connect() == "Connected to Ollama version"
    
    def test_mock_identification_mode(self, mock_llama_strategy, sample_medical_records):
        """Test que verifica el modo de identificación"""
        mock_llama_strategy.set_anonymization_mode("identify")
        assert mock_llama_strategy.anonymization_mode == "identify"
        #first test the identify method
        text1 = sample_medical_records[0]["text"]
        result = mock_llama_strategy.identify(text1)
        assert result == text1

    def test_local_identification_mode(self, llama_strategy, sample_medical_records):
        """Test que verifica el modo de identificación"""
        llama_strategy.set_anonymization_mode("identify")
        assert llama_strategy.anonymization_mode == "identify"
        #first test the identify method
        text1 = sample_medical_records[0]["text"]
        ground_truth1 = sample_medical_records[0]["anonymized_text"]
        result = llama_strategy.identify(text1)
        assert result == ground_truth1

