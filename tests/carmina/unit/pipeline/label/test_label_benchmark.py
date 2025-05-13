import pytest
import logging

logger = logging.getLogger(__name__)

from unittest.mock import MagicMock, patch
from src.carmina.llm.factory import LLMFactory
from src.carmina.pipeline.anon_pipeline import AnonymizationPipeline
from src.carmina.metrics.recorder import MetricsRecorder
from src.carmina.metrics.evaluator import evaluate_label

@pytest.mark.label
class TestLabelBenchmark:

    @pytest.fixture
    def mock_label_strategy(self):
        """Fixture that provides a LLM strategy in label mode"""
        strategy = LLMFactory.create(
            model_name="mock",
            cloud_provider="mock",
            strategy_kwargs={"anonymization_mode": "label"},
            provider_kwargs={}
        )
        return strategy
    
    def test_label_benchmark(self, mock_label_strategy, sample_medical_records):
        # Arrange
        pipeline = AnonymizationPipeline(mock_label_strategy)
        
        # Act
        results = pipeline.run([sample_medical_records[0]])

        logger.info("Results: %s", results[0]["identified_text"])
        logger.info("Results: %s", results[0]["anonymized_text"])
        # Assert
        assert pipeline.anonymization_mode == "label"
        assert results[0]["id"] == sample_medical_records[0]["id"]
        assert results[0]["anonymized_text"] == sample_medical_records[0]["anonymized_text"]

    def test_recorder_and_evaluator(self, mock_label_strategy, sample_medical_records):
        # Arrange
        pipeline = AnonymizationPipeline(mock_label_strategy)
        recorder = MetricsRecorder(model_name="mock")

        # Act
        records = pipeline.run([sample_medical_records[0]])
        
        # Extract texts and labels from ground truth and prediction records
        ground_truth_entities = sample_medical_records
        prediction_entities = records
        
        # Extract the required lists for the new function signature
        ground_truth_texts = [entity["anonymized_text"] for entity in ground_truth_entities]
        prediction_texts = [entity["anonymized_text"] for entity in prediction_entities]
        ground_truth_labels = [entity["entities_anonymized"] for entity in ground_truth_entities]
        prediction_labels = [entity["entities_anonymized"] for entity in prediction_entities]
        
        logger.info("Ground Truth Texts: %s", ground_truth_texts)
        logger.info("Ground Truth Labels: %s", ground_truth_labels)
        logger.info("Prediction Texts: %s", prediction_texts)
        logger.info("Prediction Labels: %s", prediction_labels)
        
        recorder.record_all(
            evaluate_label(
                ground_truth_texts=ground_truth_texts,
                prediction_texts=prediction_texts,
                ground_truth_labels=ground_truth_labels,
                prediction_labels=prediction_labels
            )
        )
        
        assert recorder.current_metrics["label_precision"] == 1.0
        assert recorder.current_metrics["label_recall"] == 1.0
        assert recorder.current_metrics["label_f1"] == 1.0
        assert recorder.current_metrics["label_tp"] == 1
        assert recorder.current_metrics["label_fp"] == 0
        assert recorder.current_metrics["label_fn"] == 0
        assert int(recorder.current_metrics["label_cosine_sim"]) == 1
        assert recorder.current_metrics["label_levenshtein"] == 0.0
        assert recorder.current_metrics["label_inv_levenshtein"] == 1.0
        assert recorder.current_metrics["label_overall"] == 5