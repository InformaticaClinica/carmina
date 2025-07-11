"""
Unit tests for MetricsRecorder token and cost tracking functionality.
"""
import pytest
from src.carmina.metrics.recorder import MetricsRecorder


@pytest.mark.unit
class TestMetricsRecorderTokens:
    """Test MetricsRecorder token and cost tracking."""
    
    def test_record_token_metrics(self):
        """Test recording token metrics."""
        recorder = MetricsRecorder("test-model")
        
        token_counts = {
            "system": 100,
            "user": 200,
            "total": 300
        }
        
        recorder.record_token_metrics(token_counts, "identify")
        metrics = recorder.get_current_metrics()
        
        assert "identify_tokens_system" in metrics
        assert "identify_tokens_user" in metrics
        assert "identify_tokens_total" in metrics
        assert metrics["identify_tokens_system"] == 100
        assert metrics["identify_tokens_user"] == 200
        assert metrics["identify_tokens_total"] == 300
    
    def test_record_token_metrics_without_mode(self):
        """Test recording token metrics without anonymization mode."""
        recorder = MetricsRecorder("test-model")
        
        token_counts = {
            "system": 50,
            "user": 75,
            "total": 125
        }
        
        recorder.record_token_metrics(token_counts)
        metrics = recorder.get_current_metrics()
        
        assert "tokens_system" in metrics
        assert "tokens_user" in metrics
        assert "tokens_total" in metrics
        assert metrics["tokens_system"] == 50
        assert metrics["tokens_user"] == 75
        assert metrics["tokens_total"] == 125
    
    def test_record_cost_metrics(self):
        """Test recording cost metrics."""
        recorder = MetricsRecorder("test-model")
        
        recorder.record_cost_metrics(input_cost=0.05, output_cost=0.10)
        metrics = recorder.get_current_metrics()
        
        assert "cost_input" in metrics
        assert "cost_output" in metrics
        assert "cost_total" in metrics
        assert metrics["cost_input"] == 0.05
        assert metrics["cost_output"] == 0.10
        assert abs(metrics["cost_total"] - 0.15) < 1e-10
    
    def test_record_cost_metrics_with_total(self):
        """Test recording cost metrics with explicit total."""
        recorder = MetricsRecorder("test-model")
        
        recorder.record_cost_metrics(input_cost=0.05, output_cost=0.10, total_cost=0.20)
        metrics = recorder.get_current_metrics()
        
        assert metrics["cost_input"] == 0.05
        assert metrics["cost_output"] == 0.10
        assert metrics["cost_total"] == 0.20
    
    def test_method_chaining(self):
        """Test that token and cost recording methods support chaining."""
        recorder = MetricsRecorder("test-model")
        
        token_counts = {"system": 100, "user": 200, "total": 300}
        
        # Test method chaining
        result = (recorder
                 .record_token_metrics(token_counts, "identify")
                 .record_cost_metrics(0.05, 0.10)
                 .record("additional_metric", "test_value"))
        
        # Should return the same recorder instance
        assert result is recorder
        
        # Check all metrics were recorded
        metrics = recorder.get_current_metrics()
        assert "identify_tokens_total" in metrics
        assert "cost_total" in metrics
        assert "additional_metric" in metrics
    
    def test_multiple_mode_token_tracking(self):
        """Test tracking tokens for multiple anonymization modes."""
        recorder = MetricsRecorder("test-model")
        
        # Record tokens for different modes
        identify_tokens = {"system": 100, "user": 150, "total": 250}
        label_tokens = {"system": 120, "user": 180, "total": 300}
        substitute_tokens = {"system": 110, "user": 160, "total": 270}
        
        recorder.record_token_metrics(identify_tokens, "identify")
        recorder.record_token_metrics(label_tokens, "label") 
        recorder.record_token_metrics(substitute_tokens, "substitute")
        
        metrics = recorder.get_current_metrics()
        
        # Check all modes are tracked separately
        assert metrics["identify_tokens_total"] == 250
        assert metrics["label_tokens_total"] == 300
        assert metrics["substitute_tokens_total"] == 270
        
        # Check individual components
        assert metrics["identify_tokens_system"] == 100
        assert metrics["label_tokens_user"] == 180
        assert metrics["substitute_tokens_system"] == 110
    
    def test_token_metrics_with_missing_keys(self):
        """Test token metrics recording with missing dictionary keys."""
        recorder = MetricsRecorder("test-model")
        
        # Incomplete token counts
        incomplete_counts = {"system": 100}  # Missing user and total
        
        recorder.record_token_metrics(incomplete_counts, "test")
        metrics = recorder.get_current_metrics()
        
        assert metrics["test_tokens_system"] == 100
        assert metrics["test_tokens_user"] == 0  # Should default to 0
        assert metrics["test_tokens_total"] == 0  # Should default to 0
    
    def test_integration_with_existing_metrics(self):
        """Test that token/cost metrics integrate with existing functionality."""
        recorder = MetricsRecorder("test-model")
        
        # Record traditional metrics
        recorder.record("precision", 0.95)
        recorder.record("recall", 0.90)
        
        # Record token metrics
        token_counts = {"system": 50, "user": 100, "total": 150}
        recorder.record_token_metrics(token_counts, "identify")
        
        # Record cost metrics
        recorder.record_cost_metrics(0.03, 0.06)
        
        # Save to results
        recorder.save_current()
        
        # Check current metrics
        current = recorder.get_current_metrics()
        assert len(current) == 0  # Should be cleared after save
        
        # Check saved results
        results = recorder.get_all_results()
        assert len(results) == 1
        
        saved_metrics = results[0]
        assert saved_metrics["precision"] == 0.95
        assert saved_metrics["recall"] == 0.90
        assert saved_metrics["identify_tokens_total"] == 150
        assert abs(saved_metrics["cost_total"] - 0.09) < 1e-10