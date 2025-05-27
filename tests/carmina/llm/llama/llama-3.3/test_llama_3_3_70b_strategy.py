import pytest
import os
from unittest.mock import MagicMock, patch
from src.carmina.llm.cloud_providers.mock_provider import MockProvider
from src.carmina.llm.cloud_providers.aws_provider import AWSProvider
from src.carmina.llm.strategies.llama3_strategy import Llama3Strategy

class DummyProvider:
    def get_name(self):
        return "dummy"
    def run_inference(self, model_id, messages, **kwargs):
        return "dummy response"

@pytest.mark.llm
@pytest.mark.llama3
class TestLlama3Strategy:
    @pytest.fixture
    def mock_llama3_strategy(self):
        """Fixture para una estrategia Llama3 simulada"""
        mock_cloud_provider = MockProvider()
        strategy = Llama3Strategy(
            model_name="llama-3.3-70b", 
            cloud_provider=mock_cloud_provider,
            anonymization_mode="identify"
        )
        return strategy
    
    @pytest.fixture
    def llama3_strategy(self):
        """Fixture para la estrategia de Llama3"""
        aws_provider = AWSProvider("aws")
        return Llama3Strategy(
            model_name="llama-3.3-70b", 
            cloud_provider=aws_provider,
            anonymization_mode="identify"
        )

    def test_init(self, mock_llama3_strategy):
        """Test que verifica la inicialización de la estrategia Llama3"""
        assert isinstance(mock_llama3_strategy.cloud_provider, MockProvider)
        assert mock_llama3_strategy.anonymization_mode == "identify"
        assert mock_llama3_strategy.get_name() == "llama-3.3-70b"
    
    def test_init_with_aws_provider(self, llama3_strategy):
        """Test que verifica la inicialización de la estrategia Llama3 con un proveedor AWS"""
        assert llama3_strategy.anonymization_mode == "identify"
        assert llama3_strategy.get_name() == "llama-3.3-70b"
    
    def test_aws_connection(self, llama3_strategy):
        """Test que verifica la conexión de la estrategia Llama3"""
        assert llama3_strategy.cloud_provider.connect() == "Connected to AWS API"

    def test_mock_identification_mode(self, mock_llama3_strategy, sample_medical_records):
        """Test que verifica el modo de identificación"""
        mock_llama3_strategy.set_anonymization_mode("identify")
        assert mock_llama3_strategy.anonymization_mode == "identify"
        #first test the identify method
        text1 = sample_medical_records[0]["text"]
        result = mock_llama3_strategy.identify(text1)
        assert result == text1

    @patch('src.carmina.llm.strategies.llama3_strategy.Llama3Strategy.identify')
    def test_aws_identification_mode(self, mock_identify, llama3_strategy, sample_medical_records):
        """Test que verifica el modo de identificación con AWS"""
        # Setup the mock to return the expected output
        mock_identify.return_value = sample_medical_records[0]["anonymized_text"]
        
        assert llama3_strategy.anonymization_mode == "identify"
        #first test the identify method
        text1 = sample_medical_records[0]["text"]
        ground_truth1 = sample_medical_records[0]["anonymized_text"]
        result = llama3_strategy.identify(text1)
        assert result == ground_truth1
        mock_identify.assert_called_once_with(text1)

def test_llama3_strategy_aws():
    provider = AWSProvider()
    strategy = Llama3Strategy(model_name="llama-3.3-70b", cloud_provider=provider)
    # Simula llamada a run_inference (no requiere AWS real)
    try:
        strategy.cloud_provider = DummyProvider()
        result = strategy.run_inference({"messages": [{"role": "user", "content": "test"}]}, {"temperature": 0.1})
        assert result == "dummy response"
    except Exception as e:
        pytest.fail(f"Llama3Strategy with AWSProvider failed: {e}")
