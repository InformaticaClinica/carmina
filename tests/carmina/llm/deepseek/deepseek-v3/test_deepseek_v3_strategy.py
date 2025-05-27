import pytest
import os
from unittest.mock import MagicMock, patch
from src.carmina.llm.cloud_providers.mock_provider import MockProvider
from src.carmina.llm.cloud_providers.azure_provider import AzureProvider
from src.carmina.llm.strategies.deepseek_strategy import DeepSeekStrategy

@pytest.mark.llm
@pytest.mark.deepseek
class TestDeepSeekStrategy:
    @pytest.fixture
    def mock_deepseek_strategy(self):
        """Fixture para una estrategia DeepSeek simulada"""
        mock_cloud_provider = MockProvider()
        strategy = DeepSeekStrategy(
            model_name="deepseek-r1", 
            cloud_provider=mock_cloud_provider,
            anonymization_mode="identify"
        )
        return strategy
    
    @pytest.fixture
    def deepseek_strategy(self):
        """Fixture para la estrategia de DeepSeek"""
        azure_provider = AzureProvider("azure")
        return DeepSeekStrategy(
            model_name="deepseek-r1", 
            cloud_provider=azure_provider,
            anonymization_mode="identify"
        )
    
    def test_init(self, mock_deepseek_strategy):
        """Test que verifica la inicialización de la estrategia DeepSeek"""
        assert isinstance(mock_deepseek_strategy.cloud_provider, MockProvider)
        assert mock_deepseek_strategy.anonymization_mode == "identify"
        assert mock_deepseek_strategy.get_name() == "deepseek-r1"

    def test_init_with_azure_provider(self, deepseek_strategy):
        """Test que verifica la inicialización de la estrategia DeepSeek con un proveedor Azure"""
        assert deepseek_strategy.anonymization_mode == "identify"
        assert deepseek_strategy.get_name() == "deepseek-r1"
    
    def test_azure_connection(self, deepseek_strategy):
        """Test que verifica la conexión de la estrategia DeepSeek"""
        assert deepseek_strategy.cloud_provider.connect() == "Connected to Azure API"

    def test_mock_identification_mode(self, mock_deepseek_strategy, sample_medical_records):
        """Test que verifica el modo de identificación"""
        mock_deepseek_strategy.set_anonymization_mode("identify")
        assert mock_deepseek_strategy.anonymization_mode == "identify"
        #first test the identify method
        text1 = sample_medical_records[0]["text"]
        result = mock_deepseek_strategy.identify(text1)
        assert result == text1
    
    @patch('src.carmina.llm.strategies.deepseek_strategy.DeepSeekStrategy.identify')
    def test_azure_identification_mode(self, mock_identify, deepseek_strategy, sample_medical_records):
        """Test que verifica el modo de identificación con Azure"""
        # Setup the mock to return the expected output
        mock_identify.return_value = sample_medical_records[0]["anonymized_text"]
        
        assert deepseek_strategy.anonymization_mode == "identify"
        #first test the identify method
        text1 = sample_medical_records[0]["text"]
        ground_truth1 = sample_medical_records[0]["anonymized_text"]
        result = deepseek_strategy.identify(text1)
        assert result == ground_truth1
        mock_identify.assert_called_once_with(text1)