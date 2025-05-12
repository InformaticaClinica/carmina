import pytest
import os
from unittest.mock import MagicMock, patch
from src.carmina.llm.cloud_providers.mock_provider import MockProvider
from src.carmina.llm.cloud_providers.aws_provider import AWSProvider
from src.carmina.llm.strategies.anthropic_strategy import AnthropicStrategy

@pytest.mark.llm
@pytest.mark.anthropic
@pytest.mark.sonnet
class TestClaudeSonnet3_7Strategy:
    @pytest.fixture
    def mock_anthropic_strategy(self):
        """Fixture para una estrategia Anthropic simulada"""
        mock_cloud_provider = MockProvider()
        strategy = AnthropicStrategy(model_name="claude-3.7-sonnet", cloud_provider=mock_cloud_provider)
        return strategy
    
    @pytest.fixture
    def anthropic_strategy(self):
        """Fixture para la estrategia de Anthropic"""
        aws_provider = AWSProvider("aws")
        return AnthropicStrategy(model_name="claude-3.7-sonnet", cloud_provider=aws_provider)

    def test_init(self, mock_anthropic_strategy):
        """Test que verifica la inicialización de la estrategia Anthropic"""
        assert isinstance(mock_anthropic_strategy.cloud_provider, MockProvider)
        assert mock_anthropic_strategy.anonymization_mode == "identify"
        assert mock_anthropic_strategy.get_name() == "claude-3.7-sonnet"
    
    def test_init_with_aws_provider(self, anthropic_strategy):
        """Test que verifica la inicialización de la estrategia Anthropic con un proveedor AWS"""
        assert anthropic_strategy.anonymization_mode == "identify"
        assert anthropic_strategy.get_name() == "claude-3.7-sonnet"
    
    def test_aws_connection(self, anthropic_strategy):
        """Test que verifica la conexión de la estrategia Anthropic"""
        assert anthropic_strategy.cloud_provider.connect() == "Connected to AWS API"
    
    def test_mock_identification_mode(self, mock_anthropic_strategy, sample_medical_records):
        """Test que verifica el modo de identificación"""
        mock_anthropic_strategy.set_anonymization_mode("identify")
        assert mock_anthropic_strategy.anonymization_mode == "identify"
        #first test the identify method
        text1 = sample_medical_records[0]["text"]
        result = mock_anthropic_strategy.identify(text1)
        assert result == text1

    def test_aws_identification_mode(self, anthropic_strategy, sample_medical_records):
        """Test que verifica el modo de identificación con AWS"""
        assert anthropic_strategy.anonymization_mode == "identify"
        #first test the identify method
        text1 = sample_medical_records[0]["text"]
        ground_truth1 = sample_medical_records[0]["anonymized_text"]
        result = anthropic_strategy.identify(text1)
        assert result == ground_truth1
