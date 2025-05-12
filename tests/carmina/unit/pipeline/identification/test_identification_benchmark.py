import pytest
from unittest.mock import MagicMock, patch
from src.carmina.llm.strategies.mock_strategy import MockLLMStrategy
from src.carmina.pipeline.anon_pipeline import AnonymizationPipeline
from src.carmina.metrics.recorder import MetricsRecorder
from src.carmina.metrics.evaluator import evaluate_identification

class TestIdentificationBenchmark:
    """Tests para el benchmark del pipeline en modo identificación"""
    
    @pytest.fixture
    def mock_identify_strategy(self):
        """Fixture que proporciona una estrategia LLM en modo identification"""
        strategy = MockLLMStrategy()
        return strategy
    
    def test_identification_benchmark(self, mock_identify_strategy, sample_medical_records):
        # Arrange
        pipeline = AnonymizationPipeline(mock_identify_strategy)
        
        # Act
        results = pipeline.run([sample_medical_records[0]])  # Usar el primer registro
        print("Results:", results)

        # Assert
        assert len(results) == 1
        assert results[0]["id"] == sample_medical_records[0]["id"]
        assert results[0]["anonymized_text"] == sample_medical_records[0]["anonymized_text"]
    
    def test_recorder_and_evaluator(self, mock_identify_strategy, sample_medical_records):
        # Arrange
        pipeline = AnonymizationPipeline(mock_identify_strategy)
        recorder = MetricsRecorder(model_name="mock")

        #Act
        records = pipeline.run([sample_medical_records[0]])
        ground_truth_records = sample_medical_records[0]["entities_identified"]
        prediction_records = records[0]["entities_identified"]
        print("Ground Truth Records:", ground_truth_records)
        print("Prediction Records:", prediction_records)
        recorder.record_all(
            evaluate_identification(ground_truth_records=ground_truth_records, 
                                      prediction_records=prediction_records)
        )
        assert recorder.current_metrics["identification_precision"] == 1.0
        assert recorder.current_metrics["identification_recall"] == 1.0
        assert recorder.current_metrics["identification_f1"] == 1.0
        assert recorder.current_metrics["identification_tp"] == 1
        assert recorder.current_metrics["identification_fp"] == 0
        assert recorder.current_metrics["identification_fn"] == 0

    def test_bad_evaluator_and_recorder(self, sample_medical_records):
        # Arrange
        recorder = MetricsRecorder(model_name="mock")

        #Act
        ground_truth_records = sample_medical_records[0]["entities_identified"]
        prediction_records = ["**Mock**"]
        recorder.record_all(
            evaluate_identification(ground_truth_records=ground_truth_records, 
                                      prediction_records=prediction_records)
        )
        assert recorder.current_metrics["identification_precision"] == 0
        assert recorder.current_metrics["identification_recall"] == 0
        assert recorder.current_metrics["identification_f1"] == 0
        assert recorder.current_metrics["identification_tp"] == 0
        assert recorder.current_metrics["identification_fp"] == 1
        assert recorder.current_metrics["identification_fn"] == 1

    def test_good_and_bad_recorder(self, sample_medical_records):
        # Arrange
        recorder = MetricsRecorder(model_name="mock")

        #Act
        ground_truth_records = sample_medical_records[0]["entities_identified"]
        prediction_records = ["**Mock**", "Juan García"]
        recorder.record_all(
            evaluate_identification(ground_truth_records=ground_truth_records, 
                                      prediction_records=prediction_records)
        )
        assert recorder.current_metrics["identification_precision"] == 0.5
        assert recorder.current_metrics["identification_recall"] == 1
        assert recorder.current_metrics["identification_f1"] == 0.6666666666666666
        assert recorder.current_metrics["identification_tp"] == 1
        assert recorder.current_metrics["identification_fp"] == 1
        assert recorder.current_metrics["identification_fn"] == 0