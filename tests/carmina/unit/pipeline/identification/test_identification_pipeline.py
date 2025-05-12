import pytest
from unittest.mock import MagicMock, patch
from src.carmina.llm.strategies.mock_strategy import MockLLMStrategy
from src.carmina.pipeline.anon_pipeline import AnonymizationPipeline

class TestIdentificationPipeline:
    """Tests para el pipeline en modo identificación"""
    
    @pytest.fixture
    def mock_identify_strategy(self):
        """Fixture que proporciona una estrategia LLM en modo identification"""
        strategy = MockLLMStrategy()
        return strategy
    
    def test_identification_pipeline_init(self, mock_identify_strategy):
        """Test que el pipeline se inicializa correctamente en modo identify"""
        # Arrange & Act
        pipeline = AnonymizationPipeline(mock_identify_strategy)
        
        # Assert
        assert pipeline.anonymization_mode == "identify"
        assert pipeline.identification is not None
        assert pipeline.anonymizer is None
    
    def test_identification_pipeline_flow(self, mock_identify_strategy, sample_medical_records):
        # Arrange
        pipeline = AnonymizationPipeline(mock_identify_strategy)
        
        # Act
        results = pipeline.run([sample_medical_records[0]])  # Usar el primer registro
        print("Results:", results)

        # Assert
        assert len(results) == 1
        assert results[0]["id"] == sample_medical_records[0]["id"]
        assert results[0]["anonymized_text"] == sample_medical_records[0]["anonymized_text"]