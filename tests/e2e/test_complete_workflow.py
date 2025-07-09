"""
End-to-end tests for complete anonymization workflows.
"""
import pytest
import json
import os
from unittest.mock import patch, MagicMock
from src.carmina.tools.benchmark_runner import BenchmarkRunner
from src.carmina.model_executor import ModelExecutor
from src.carmina.llm.factory import LLMFactory


@pytest.mark.e2e
class TestCompleteWorkflow:
    """End-to-end tests for complete anonymization workflows."""
    
    @pytest.fixture
    def sample_input_data(self, tmp_path):
        """Create sample input data for testing."""
        input_data = [
            {
                "id": "CARMEN-I_CC_1",
                "text": "El paciente Juan García fue atendido por el Dr. Martínez el 12/05/2023.",
                "identify": "El paciente [**Juan García**] fue atendido por el Dr. [**Martínez**] el [**12/05/2023**].",
                "masked_text": "El paciente [**OTROS_SUJETO_ASISTENCIA**] fue atendido por el Dr. [**NOMBRE_PERSONAL_SANITARIO**] el [**FECHAS**]."
            },
            {
                "id": "CARMEN-I_CC_2",
                "text": "María López presenta dolor abdominal desde hace 3 días.",
                "identify": "[**María López**] presenta dolor abdominal desde hace [**3 días**].",
                "masked_text": "[**OTROS_SUJETO_ASISTENCIA**] presenta dolor abdominal desde hace [**FECHAS**]."
            }
        ]
        
        input_file = tmp_path / "input.json"
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(input_data, f, ensure_ascii=False, indent=2)
        
        return str(input_file)
    
    @pytest.fixture
    def test_directories(self, tmp_path):
        """Create test directories."""
        dirs = {
            'output': tmp_path / "outputs",
            'metrics': tmp_path / "metrics",
            'debug': tmp_path / "debug"
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        return {k: str(v) for k, v in dirs.items()}
    
    @patch('src.carmina.data_sources.loaders.load_json_file')
    @patch('src.carmina.metrics.compare_line.extract_all_metrics')
    def test_complete_identify_workflow(self, mock_extract_metrics, mock_load_json, 
                                      sample_input_data, test_directories):
        """Test complete identify workflow from input to output."""
        # Mock data loading
        mock_load_json.return_value = [
            {
                "id": "CARMEN-I_CC_1",
                "text": "El paciente Juan García fue atendido.",
                "identify": "El paciente [**Juan García**] fue atendido.",
                "masked_text": "El paciente [**OTROS_SUJETO_ASISTENCIA**] fue atendido."
            }
        ]
        
        # Mock metrics extraction
        mock_extract_metrics.return_value = {
            "identification_precision": 0.9,
            "identification_recall": 0.8,
            "identification_f1": 0.85,
            "label_precision": 0.7,
            "label_recall": 0.8,
            "label_f1": 0.75
        }
        
        # Create model executor
        executor = ModelExecutor(
            model_name="mock-model",
            anonymization_mode="identify",
            cloud_provider="mock",
            input_path=sample_input_data,
            output_dir=test_directories['output'],
            metrics_dir=test_directories['metrics'],
            debug_dir=test_directories['debug']
        )
        
        # Execute the workflow
        with patch.dict(os.environ, {
            'OUTPUT_DIR': test_directories['output']
        }):
            executor.execute()
        
        # Verify results
        mock_load_json.assert_called_once()
        mock_extract_metrics.assert_called_once()
    
    @patch('src.carmina.data_sources.loaders.load_json_file')
    @patch('src.carmina.metrics.compare_line.extract_all_metrics')
    def test_complete_label_workflow(self, mock_extract_metrics, mock_load_json, 
                                   sample_input_data, test_directories):
        """Test complete label workflow from input to output."""
        # Mock data loading
        mock_load_json.return_value = [
            {
                "id": "CARMEN-I_CC_1",
                "text": "El paciente Juan García fue atendido.",
                "identify": "El paciente [**Juan García**] fue atendido.",
                "masked_text": "El paciente [**OTROS_SUJETO_ASISTENCIA**] fue atendido."
            }
        ]
        
        # Mock metrics extraction
        mock_extract_metrics.return_value = {
            "identification_precision": 0.9,
            "identification_recall": 0.8,
            "identification_f1": 0.85,
            "label_precision": 0.85,
            "label_recall": 0.9,
            "label_f1": 0.875
        }
        
        # Create model executor
        executor = ModelExecutor(
            model_name="mock-model",
            anonymization_mode="label",
            cloud_provider="mock",
            input_path=sample_input_data,
            output_dir=test_directories['output'],
            metrics_dir=test_directories['metrics'],
            debug_dir=test_directories['debug']
        )
        
        # Execute the workflow
        with patch.dict(os.environ, {
            'OUTPUT_DIR': test_directories['output']
        }):
            executor.execute()
        
        # Verify results
        mock_load_json.assert_called_once()
        mock_extract_metrics.assert_called_once()
    
    @patch('src.carmina.data_sources.loaders.load_json_file')
    @patch('src.carmina.metrics.compare_line.extract_all_metrics')
    def test_complete_substitute_workflow(self, mock_extract_metrics, mock_load_json, 
                                        sample_input_data, test_directories):
        """Test complete substitute workflow from input to output."""
        # Mock data loading
        mock_load_json.return_value = [
            {
                "id": "CARMEN-I_CC_1",
                "text": "El paciente Juan García fue atendido.",
                "identify": "El paciente [**Juan García**] fue atendido.",
                "masked_text": "El paciente [**OTROS_SUJETO_ASISTENCIA**] fue atendido."
            }
        ]
        
        # Mock metrics extraction
        mock_extract_metrics.return_value = {
            "identification_precision": 0.9,
            "identification_recall": 0.8,
            "identification_f1": 0.85,
            "label_precision": 0.8,
            "label_recall": 0.75,
            "label_f1": 0.775
        }
        
        # Create model executor
        executor = ModelExecutor(
            model_name="mock-model",
            anonymization_mode="substitute",
            cloud_provider="mock",
            input_path=sample_input_data,
            output_dir=test_directories['output'],
            metrics_dir=test_directories['metrics'],
            debug_dir=test_directories['debug']
        )
        
        # Execute the workflow
        with patch.dict(os.environ, {
            'OUTPUT_DIR': test_directories['output']
        }):
            executor.execute()
        
        # Verify results
        mock_load_json.assert_called_once()
        mock_extract_metrics.assert_called_once()
    
    @patch('src.carmina.data_sources.loaders.load_json_file')
    @patch('src.carmina.metrics.compare_line.extract_all_metrics')
    @patch('src.carmina.tools.benchmark_summary.BenchmarkSummary')
    def test_complete_benchmark_workflow(self, mock_summary, mock_extract_metrics, 
                                       mock_load_json, sample_input_data, test_directories):
        """Test complete benchmark workflow with multiple models."""
        # Mock data loading
        mock_load_json.return_value = [
            {
                "id": "CARMEN-I_CC_1",
                "text": "El paciente Juan García fue atendido.",
                "identify": "El paciente [**Juan García**] fue atendido.",
                "masked_text": "El paciente [**OTROS_SUJETO_ASISTENCIA**] fue atendido."
            }
        ]
        
        # Mock metrics extraction
        mock_extract_metrics.return_value = {
            "identification_precision": 0.9,
            "identification_recall": 0.8,
            "identification_f1": 0.85,
            "label_precision": 0.85,
            "label_recall": 0.9,
            "label_f1": 0.875
        }
        
        # Mock benchmark summary
        mock_summary_instance = MagicMock()
        mock_summary.return_value = mock_summary_instance
        
        # Run benchmark
        with patch.dict(os.environ, {
            'MODELS': 'mock-model1,mock-model2',
            'ANONYMIZATION_MODE': 'identify',
            'CLOUD_PROVIDER': 'mock',
            'INPUT_DIR': sample_input_data,
            'OUTPUT_DIR': test_directories['output'],
            'METRICS_DIR': test_directories['metrics'],
            'DEBUG_DIR': test_directories['debug']
        }):
            runner = BenchmarkRunner()
            runner.run()
        
        # Verify benchmark summary was generated
        mock_summary.assert_called_once_with(['mock-model1', 'mock-model2'], test_directories['metrics'])
        mock_summary_instance.generate.assert_called_once()
    
    def test_llm_factory_integration(self):
        """Test LLM factory integration with different configurations."""
        # Test different model configurations
        test_configs = [
            {"model": "mock-model", "provider": "mock", "mode": "identify"},
            {"model": "mock-model", "provider": "mock", "mode": "label"},
            {"model": "mock-model", "provider": "mock", "mode": "substitute"},
            {"model": "mock-model", "provider": "local", "mode": "identify"},
        ]
        
        for config in test_configs:
            llm = LLMFactory.create(
                model_name=config["model"],
                cloud_provider=config["provider"],
                strategy_kwargs={"anonymization_mode": config["mode"]}
            )
            
            assert llm is not None
            assert llm.model_name == config["model"]
            assert llm.anonymization_mode == config["mode"]
    
    def test_pipeline_integration_with_factory(self):
        """Test pipeline integration with LLM factory."""
        from src.carmina.pipeline.anon_pipeline import AnonymizationPipeline
        
        # Create LLM through factory
        llm = LLMFactory.create(
            model_name="mock-model",
            cloud_provider="mock",
            strategy_kwargs={"anonymization_mode": "identify"}
        )
        
        # Create pipeline
        pipeline = AnonymizationPipeline(llm)
        
        # Test pipeline configuration
        assert pipeline.anonymization_mode == "identify"
        assert pipeline.llm_strategy == llm
        assert pipeline.identification is not None
        assert pipeline.anonymizer is None
    
    def test_error_handling_in_complete_workflow(self, sample_input_data, test_directories):
        """Test error handling in complete workflow."""
        # Create executor with invalid configuration
        executor = ModelExecutor(
            model_name="invalid-model",
            anonymization_mode="invalid-mode",
            cloud_provider="invalid-provider",
            input_path=sample_input_data,
            output_dir=test_directories['output'],
            metrics_dir=test_directories['metrics'],
            debug_dir=test_directories['debug']
        )
        
        # Should handle errors gracefully
        with patch.dict(os.environ, {
            'OUTPUT_DIR': test_directories['output']
        }):
            try:
                executor.execute()
            except Exception as e:
                # Should fail gracefully with meaningful error
                assert isinstance(e, (ValueError, KeyError, ImportError))
    
    @patch('src.carmina.data_sources.loaders.load_json_file')
    def test_empty_input_handling(self, mock_load_json, sample_input_data, test_directories):
        """Test handling of empty input data."""
        # Mock empty data loading
        mock_load_json.return_value = []
        
        # Create model executor
        executor = ModelExecutor(
            model_name="mock-model",
            anonymization_mode="identify",
            cloud_provider="mock",
            input_path=sample_input_data,
            output_dir=test_directories['output'],
            metrics_dir=test_directories['metrics'],
            debug_dir=test_directories['debug']
        )
        
        # Execute the workflow with empty data
        with patch.dict(os.environ, {
            'OUTPUT_DIR': test_directories['output']
        }):
            executor.execute()
        
        # Should handle empty input gracefully
        mock_load_json.assert_called_once()
    
    @patch('src.carmina.data_sources.loaders.load_json_file')
    def test_malformed_input_handling(self, mock_load_json, sample_input_data, test_directories):
        """Test handling of malformed input data."""
        # Mock malformed data loading
        mock_load_json.return_value = [
            {"id": "test1"},  # Missing text field
            {"text": "No id field"},  # Missing id field
            {"id": "test3", "text": ""}  # Empty text
        ]
        
        # Create model executor
        executor = ModelExecutor(
            model_name="mock-model",
            anonymization_mode="identify",
            cloud_provider="mock",
            input_path=sample_input_data,
            output_dir=test_directories['output'],
            metrics_dir=test_directories['metrics'],
            debug_dir=test_directories['debug']
        )
        
        # Execute the workflow with malformed data
        with patch.dict(os.environ, {
            'OUTPUT_DIR': test_directories['output']
        }):
            executor.execute()
        
        # Should handle malformed input gracefully
        mock_load_json.assert_called_once()