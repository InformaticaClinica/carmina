import pytest
from unittest.mock import MagicMock, patch
from src.carmina.llm.strategies.mock_strategy import MockLLMStrategy
from src.carmina.pipeline.anon_pipeline import AnonymizationPipeline

class TestLabelPipeline:
    """Pipeline tests for the label mode"""

    @pytest.fixture
    def mock_label_strategy(self):
        """Fixture that provides a LLM strategy in label mode"""
        strategy = MockLLMStrategy(anonymization_mode="label")
        return strategy
    
    def test_label_pipeline_init(self, mock_label_strategy):
        """Test that the pipeline initializes correctly in label mode"""
        # Arrange & Act
        pipeline = AnonymizationPipeline(mock_label_strategy)
        
        # Assert
        assert pipeline.anonymization_mode == "label"
        assert pipeline.anonymizer is not None

    def test_label_pipeline_flow(self, mock_label_strategy, sample_medical_records):
        # Arrange
        pipeline = AnonymizationPipeline(mock_label_strategy)
        
        # Act
        results = pipeline.run([sample_medical_records[0]])
        print("Results:", results)
        
        # Assert
        assert len(results) == 1
        assert results[0]["id"] == sample_medical_records[0]["id"]
        assert results[0]["anonymized_text"] == sample_medical_records[0]["anonymized_text"]