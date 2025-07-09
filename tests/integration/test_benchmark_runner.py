"""
Integration tests for benchmark runner functionality.
"""
import pytest
import os
from unittest.mock import MagicMock, patch
from src.carmina.tools.benchmark_runner import BenchmarkRunner


@pytest.mark.integration
class TestBenchmarkRunner:
    """Integration tests for BenchmarkRunner."""
    
    def test_benchmark_runner_init_defaults(self):
        """Test benchmark runner initialization with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            runner = BenchmarkRunner()
            
            assert runner.models == ["chatgpt"]
            assert runner.anonymization_mode == "label"
            assert runner.cloud_provider == "local"
            assert runner.input_path == "data/input.json"
            assert runner.output_dir == "data/outputs/"
            assert runner.metrics_dir == "metrics/"
            assert runner.debug_dir == "data/outputs/debug/"
    
    def test_benchmark_runner_init_custom_env(self):
        """Test benchmark runner initialization with custom environment."""
        with patch.dict(os.environ, {
            'MODELS': 'claude-3.5-sonnet,gpt-4-turbo',
            'ANONYMIZATION_MODE': 'identify',
            'CLOUD_PROVIDER': 'aws',
            'INPUT_DIR': 'custom/input.json',
            'OUTPUT_DIR': 'custom/outputs/',
            'METRICS_DIR': 'custom/metrics/',
            'DEBUG_DIR': 'custom/debug/'
        }):
            runner = BenchmarkRunner()
            
            assert runner.models == ["claude-3.5-sonnet", "gpt-4-turbo"]
            assert runner.anonymization_mode == "identify"
            assert runner.cloud_provider == "aws"
            assert runner.input_path == "custom/input.json"
            assert runner.output_dir == "custom/outputs/"
            assert runner.metrics_dir == "custom/metrics/"
            assert runner.debug_dir == "custom/debug/"
    
    @patch('os.makedirs')
    def test_benchmark_runner_creates_directories(self, mock_makedirs):
        """Test benchmark runner creates necessary directories."""
        with patch.dict(os.environ, {
            'OUTPUT_DIR': 'test/outputs/',
            'METRICS_DIR': 'test/metrics/',
            'DEBUG_DIR': 'test/debug/'
        }):
            runner = BenchmarkRunner()
            
            # Check that makedirs was called for each directory
            mock_makedirs.assert_any_call('test/outputs/', exist_ok=True)
            mock_makedirs.assert_any_call('test/metrics/', exist_ok=True)
            mock_makedirs.assert_any_call('test/debug/', exist_ok=True)
    
    @patch('src.carmina.tools.benchmark_runner.ModelExecutor')
    @patch('src.carmina.tools.benchmark_runner.BenchmarkSummary')
    @patch('os.makedirs')
    def test_benchmark_runner_run_single_model(self, mock_makedirs, mock_summary, mock_executor):
        """Test benchmark runner run with single model."""
        # Setup mocks
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        
        mock_summary_instance = MagicMock()
        mock_summary.return_value = mock_summary_instance
        
        with patch.dict(os.environ, {
            'MODELS': 'mock-model',
            'ANONYMIZATION_MODE': 'identify',
            'CLOUD_PROVIDER': 'mock'
        }):
            runner = BenchmarkRunner()
            runner.run()
            
            # Verify ModelExecutor was called with correct parameters
            mock_executor.assert_called_once_with(
                model_name="mock-model",
                anonymization_mode="identify",
                cloud_provider="mock",
                input_path="data/input.json",
                output_dir="data/outputs/",
                metrics_dir="metrics/",
                debug_dir="data/outputs/debug/"
            )
            
            # Verify execute was called
            mock_executor_instance.execute.assert_called_once()
            
            # Verify summary was generated
            mock_summary.assert_called_once_with(["mock-model"], "metrics/")
            mock_summary_instance.generate.assert_called_once()
    
    @patch('src.carmina.tools.benchmark_runner.ModelExecutor')
    @patch('src.carmina.tools.benchmark_runner.BenchmarkSummary')
    @patch('os.makedirs')
    def test_benchmark_runner_run_multiple_models(self, mock_makedirs, mock_summary, mock_executor):
        """Test benchmark runner run with multiple models."""
        # Setup mocks
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        
        mock_summary_instance = MagicMock()
        mock_summary.return_value = mock_summary_instance
        
        with patch.dict(os.environ, {
            'MODELS': 'model1,model2,model3',
            'ANONYMIZATION_MODE': 'label',
            'CLOUD_PROVIDER': 'mock'
        }):
            runner = BenchmarkRunner()
            runner.run()
            
            # Verify ModelExecutor was called for each model
            assert mock_executor.call_count == 3
            
            # Verify execute was called for each model
            assert mock_executor_instance.execute.call_count == 3
            
            # Verify summary was generated with all models
            mock_summary.assert_called_once_with(["model1", "model2", "model3"], "metrics/")
            mock_summary_instance.generate.assert_called_once()
    
    @patch('src.carmina.tools.benchmark_runner.ModelExecutor')
    @patch('src.carmina.tools.benchmark_runner.BenchmarkSummary')
    @patch('os.makedirs')
    def test_benchmark_runner_run_with_debug(self, mock_makedirs, mock_summary, mock_executor):
        """Test benchmark runner run with debug mode."""
        # Setup mocks
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        
        mock_summary_instance = MagicMock()
        mock_summary.return_value = mock_summary_instance
        
        with patch.dict(os.environ, {
            'MODELS': 'mock-model',
            'DEBUG': 'true',
            'DEBUG_DIR': 'test/debug/'
        }):
            runner = BenchmarkRunner()
            
            assert runner.debug == "true"
            assert runner.debug_dir == "test/debug/"
            
            runner.run()
            
            # Verify ModelExecutor was called with debug directory
            mock_executor.assert_called_once_with(
                model_name="mock-model",
                anonymization_mode="label",
                cloud_provider="local",
                input_path="data/input.json",
                output_dir="data/outputs/",
                metrics_dir="metrics/",
                debug_dir="test/debug/"
            )
    
    @patch('src.carmina.tools.benchmark_runner.ModelExecutor')
    @patch('src.carmina.tools.benchmark_runner.BenchmarkSummary')
    @patch('os.makedirs')
    def test_benchmark_runner_executor_exception(self, mock_makedirs, mock_summary, mock_executor):
        """Test benchmark runner handles executor exceptions."""
        # Setup mocks
        mock_executor_instance = MagicMock()
        mock_executor_instance.execute.side_effect = Exception("Model execution failed")
        mock_executor.return_value = mock_executor_instance
        
        mock_summary_instance = MagicMock()
        mock_summary.return_value = mock_summary_instance
        
        with patch.dict(os.environ, {
            'MODELS': 'failing-model',
            'ANONYMIZATION_MODE': 'identify'
        }):
            runner = BenchmarkRunner()
            
            # Should not raise exception, but handle it gracefully
            runner.run()
            
            # Verify executor was called
            mock_executor.assert_called_once()
            mock_executor_instance.execute.assert_called_once()
            
            # Summary should still be generated
            mock_summary.assert_called_once()
            mock_summary_instance.generate.assert_called_once()
    
    @patch('src.carmina.tools.benchmark_runner.ModelExecutor')
    @patch('src.carmina.tools.benchmark_runner.BenchmarkSummary')
    @patch('os.makedirs')
    def test_benchmark_runner_summary_exception(self, mock_makedirs, mock_summary, mock_executor):
        """Test benchmark runner handles summary exceptions."""
        # Setup mocks
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        
        mock_summary_instance = MagicMock()
        mock_summary_instance.generate.side_effect = Exception("Summary generation failed")
        mock_summary.return_value = mock_summary_instance
        
        with patch.dict(os.environ, {
            'MODELS': 'mock-model',
            'ANONYMIZATION_MODE': 'identify'
        }):
            runner = BenchmarkRunner()
            
            # Should not raise exception, but handle it gracefully
            runner.run()
            
            # Verify executor was called
            mock_executor.assert_called_once()
            mock_executor_instance.execute.assert_called_once()
            
            # Summary should be attempted
            mock_summary.assert_called_once()
            mock_summary_instance.generate.assert_called_once()
    
    def test_benchmark_runner_models_parsing(self):
        """Test benchmark runner correctly parses models string."""
        test_cases = [
            ("model1", ["model1"]),
            ("model1,model2", ["model1", "model2"]),
            ("model1,model2,model3", ["model1", "model2", "model3"]),
            ("model1, model2, model3", ["model1", " model2", " model3"]),
            ("", [""]),
        ]
        
        for models_str, expected in test_cases:
            with patch.dict(os.environ, {'MODELS': models_str}):
                runner = BenchmarkRunner()
                assert runner.models == expected
    
    def test_benchmark_runner_debug_flag_parsing(self):
        """Test benchmark runner correctly parses debug flag."""
        test_cases = [
            ("true", "true"),
            ("false", "false"),
            ("TRUE", "true"),
            ("FALSE", "false"),
            ("1", "1"),
            ("0", "0"),
            ("yes", "yes"),
            ("no", "no"),
        ]
        
        for debug_str, expected in test_cases:
            with patch.dict(os.environ, {'DEBUG': debug_str}):
                runner = BenchmarkRunner()
                assert runner.debug == expected