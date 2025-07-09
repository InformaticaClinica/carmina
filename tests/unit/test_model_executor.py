"""
Unit tests for model executor functionality.
"""
import pytest
import os
import json
from unittest.mock import MagicMock, patch, mock_open
from src.carmina.model_executor import ModelExecutor


@pytest.mark.unit
class TestModelExecutor:
    """Test model executor functionality."""
    
    def test_init(self):
        """Test model executor initialization."""
        executor = ModelExecutor(
            model_name="test-model",
            anonymization_mode="identify",
            cloud_provider="mock",
            input_path="test/input.json",
            output_dir="test/output",
            metrics_dir="test/metrics",
            debug_dir="test/debug"
        )
        
        assert executor.model_name == "test-model"
        assert executor.anonymization_mode == "identify"
        assert executor.cloud_provider == "mock"
        assert executor.input_path == "test/input.json"
        assert executor.output_dir == "test/output"
        assert executor.metrics_dir == "test/metrics"
        assert executor.debug_dir == "test/debug"
    
    @patch('src.carmina.model_executor.load_dataset')
    @patch('src.carmina.model_executor.LLMFactory')
    @patch('src.carmina.model_executor.AnonymizationPipeline')
    @patch('src.carmina.model_executor.MetricsRecorder')
    @patch('src.carmina.model_executor.extract_all_metrics')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_execute_success(self, mock_file, mock_makedirs, mock_extract_metrics,
                           mock_metrics_recorder, mock_pipeline, mock_llm_factory,
                           mock_load_dataset):
        """Test successful execution."""
        # Setup mocks
        mock_records = [
            {
                "id": "test1",
                "text": "Test text",
                "identify": "Test [**text**]",
                "masked_text": "Test [**OTROS_SUJETO_ASISTENCIA**]",
                "identified_text": "Test [**text**]",
                "anonymized_text": "Test [**OTROS_SUJETO_ASISTENCIA**]"
            }
        ]
        mock_load_dataset.return_value = mock_records
        
        mock_llm = MagicMock()
        mock_llm_factory.create.return_value = mock_llm
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.return_value = mock_records
        mock_pipeline.return_value = mock_pipeline_instance
        
        mock_recorder = MagicMock()
        mock_metrics_recorder.return_value = mock_recorder
        
        mock_extract_metrics.return_value = {"precision": 0.8, "recall": 0.9}
        
        # Create executor
        executor = ModelExecutor(
            model_name="test-model",
            anonymization_mode="identify",
            cloud_provider="mock",
            input_path="test/input.json",
            output_dir="test/output",
            metrics_dir="test/metrics",
            debug_dir="test/debug"
        )
        
        # Execute
        with patch.dict(os.environ, {'OUTPUT_DIR': 'test/output'}):
            executor.execute()
        
        # Verify calls
        mock_load_dataset.assert_called_once_with("test/input.json")
        mock_llm_factory.create.assert_called_once()
        mock_pipeline.assert_called_once_with(mock_llm)
        mock_pipeline_instance.run.assert_called_once_with(mock_records)
        mock_recorder.record_all.assert_called_once()
        mock_recorder.export_to_json.assert_called_once()
    
    @patch('src.carmina.model_executor.load_dataset')
    def test_execute_no_output_dir(self, mock_load_dataset):
        """Test execution fails without OUTPUT_DIR."""
        mock_load_dataset.return_value = []
        
        executor = ModelExecutor(
            model_name="test-model",
            anonymization_mode="identify",
            cloud_provider="mock",
            input_path="test/input.json",
            output_dir="test/output",
            metrics_dir="test/metrics",
            debug_dir="test/debug"
        )
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OUTPUT_DIR not specified"):
                executor.execute()
    
    @patch('pandas.read_csv')
    def test_get_language_success(self, mock_read_csv):
        """Test successful language retrieval."""
        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.__getitem__.return_value = mock_df  # For df['filename'] == id
        mock_df.empty = False
        mock_df.values = ['es']
        mock_read_csv.return_value = mock_df
        
        executor = ModelExecutor(
            model_name="test-model",
            anonymization_mode="identify",
            cloud_provider="mock",
            input_path="test/input.json",
            output_dir="test/output",
            metrics_dir="test/metrics",
            debug_dir="test/debug"
        )
        
        language = executor.get_language("CARMEN-I_CC_1.txt")
        assert language == 'es'
    
    @patch('pandas.read_csv')
    def test_get_language_not_found(self, mock_read_csv):
        """Test language retrieval when file not found."""
        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.__getitem__.return_value = mock_df  # For df['filename'] == id
        mock_df.empty = True
        mock_read_csv.return_value = mock_df
        
        executor = ModelExecutor(
            model_name="test-model",
            anonymization_mode="identify",
            cloud_provider="mock",
            input_path="test/input.json",
            output_dir="test/output",
            metrics_dir="test/metrics",
            debug_dir="test/debug"
        )
        
        language = executor.get_language("CARMEN-I_CC_999.txt")
        assert language is None
    
    @patch('pandas.read_csv')
    def test_get_language_exception(self, mock_read_csv):
        """Test language retrieval with exception."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        executor = ModelExecutor(
            model_name="test-model",
            anonymization_mode="identify",
            cloud_provider="mock",
            input_path="test/input.json",
            output_dir="test/output",
            metrics_dir="test/metrics",
            debug_dir="test/debug"
        )
        
        language = executor.get_language("CARMEN-I_CC_1.txt")
        assert language is None
    
    def test_get_language_removes_txt_extension(self):
        """Test that get_language removes .txt extension."""
        executor = ModelExecutor(
            model_name="test-model",
            anonymization_mode="identify",
            cloud_provider="mock",
            input_path="test/input.json",
            output_dir="test/output",
            metrics_dir="test/metrics",
            debug_dir="test/debug"
        )
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_df = MagicMock()
            mock_df.__getitem__.return_value = mock_df
            mock_df.empty = True
            mock_read_csv.return_value = mock_df
            
            executor.get_language("CARMEN-I_CC_1.txt")
            
            # Check that the method is called with the filename without .txt
            mock_df.__getitem__.assert_called_with('CARMEN-I_CC_1')