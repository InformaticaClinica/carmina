"""
Integration tests for anonymization pipeline functionality.
"""
import pytest
import os
from unittest.mock import MagicMock, patch
from src.carmina.pipeline.anon_pipeline import AnonymizationPipeline
from src.carmina.llm.strategies.mock_strategy import MockLLMStrategy


@pytest.mark.integration
class TestAnonymizationPipeline:
    """Integration tests for AnonymizationPipeline."""
    
    @pytest.fixture
    def mock_strategy_identify(self):
        """Mock strategy for identify mode."""
        strategy = MockLLMStrategy(
            model_name="mock-model",
            anonymization_mode="identify"
        )
        return strategy
    
    @pytest.fixture
    def mock_strategy_label(self):
        """Mock strategy for label mode."""
        strategy = MockLLMStrategy(
            model_name="mock-model",
            anonymization_mode="label"
        )
        return strategy
    
    @pytest.fixture
    def mock_strategy_substitute(self):
        """Mock strategy for substitute mode."""
        strategy = MockLLMStrategy(
            model_name="mock-model",
            anonymization_mode="substitute"
        )
        return strategy
    
    def test_pipeline_init_identify_mode(self, mock_strategy_identify):
        """Test pipeline initialization in identify mode."""
        pipeline = AnonymizationPipeline(mock_strategy_identify)
        
        assert pipeline.llm_strategy == mock_strategy_identify
        assert pipeline.anonymization_mode == "identify"
        assert pipeline.identification is not None
        assert pipeline.anonymizer is None
        assert pipeline.processed_count == 0
    
    def test_pipeline_init_label_mode(self, mock_strategy_label):
        """Test pipeline initialization in label mode."""
        pipeline = AnonymizationPipeline(mock_strategy_label)
        
        assert pipeline.llm_strategy == mock_strategy_label
        assert pipeline.anonymization_mode == "label"
        assert pipeline.identification is not None
        assert pipeline.anonymizer is not None
        assert pipeline.processed_count == 0
    
    def test_pipeline_init_substitute_mode(self, mock_strategy_substitute):
        """Test pipeline initialization in substitute mode."""
        pipeline = AnonymizationPipeline(mock_strategy_substitute)
        
        assert pipeline.llm_strategy == mock_strategy_substitute
        assert pipeline.anonymization_mode == "substitute"
        assert pipeline.identification is not None
        assert pipeline.anonymizer is not None
        assert pipeline.processed_count == 0
    
    def test_pipeline_init_invalid_mode(self):
        """Test pipeline initialization with invalid mode."""
        strategy = MagicMock()
        strategy.anonymization_mode = "invalid"
        
        with pytest.raises(ValueError, match="Unsupported anonymization mode"):
            AnonymizationPipeline(strategy)
    
    def test_pipeline_run_identify_mode(self, mock_strategy_identify, sample_records):
        """Test pipeline run in identify mode."""
        pipeline = AnonymizationPipeline(mock_strategy_identify)
        
        # Mock the identification processor
        with patch.object(pipeline.identification, 'process') as mock_process:
            mock_process.return_value = {
                "anonymized_text": "El paciente [**Juan García**] fue atendido.",
                "entities": ["Juan García"]
            }
            
            results = pipeline.run(sample_records[:1])
            
            assert len(results) == 1
            assert results[0]["id"] == "CARMEN-I_CC_1"
            assert results[0]["identified_text"] == "El paciente [**Juan García**] fue atendido."
            assert results[0]["anonymized_text"] == "El paciente [**Juan García**] fue atendido."
            assert pipeline.processed_count == 1
    
    def test_pipeline_run_label_mode(self, mock_strategy_label, sample_records):
        """Test pipeline run in label mode."""
        pipeline = AnonymizationPipeline(mock_strategy_label)
        
        # Mock the processors
        with patch.object(pipeline.identification, 'process') as mock_id_process, \
             patch.object(pipeline.anonymizer, 'process') as mock_anon_process:
            
            mock_id_process.return_value = {
                "anonymized_text": "El paciente [**Juan García**] fue atendido.",
                "entities": ["Juan García"]
            }
            
            mock_anon_process.return_value = {
                "anonymized_text": "El paciente [**OTROS_SUJETO_ASISTENCIA**] fue atendido.",
                "entities": ["OTROS_SUJETO_ASISTENCIA"]
            }
            
            results = pipeline.run(sample_records[:1])
            
            assert len(results) == 1
            assert results[0]["id"] == "CARMEN-I_CC_1"
            assert results[0]["identified_text"] == "El paciente [**Juan García**] fue atendido."
            assert results[0]["anonymized_text"] == "El paciente [**OTROS_SUJETO_ASISTENCIA**] fue atendido."
            assert pipeline.processed_count == 1
    
    def test_pipeline_run_substitute_mode(self, mock_strategy_substitute, sample_records):
        """Test pipeline run in substitute mode."""
        pipeline = AnonymizationPipeline(mock_strategy_substitute)
        
        # Mock the processors
        with patch.object(pipeline.identification, 'process') as mock_id_process, \
             patch.object(pipeline.anonymizer, 'process') as mock_anon_process:
            
            mock_id_process.return_value = {
                "anonymized_text": "El paciente [**Juan García**] fue atendido.",
                "entities": ["Juan García"]
            }
            
            mock_anon_process.return_value = {
                "anonymized_text": "El paciente [**Carlos Mendez**] fue atendido.",
                "entities": ["Carlos Mendez"]
            }
            
            results = pipeline.run(sample_records[:1])
            
            assert len(results) == 1
            assert results[0]["id"] == "CARMEN-I_CC_1"
            assert results[0]["identified_text"] == "El paciente [**Juan García**] fue atendido."
            assert results[0]["anonymized_text"] == "El paciente [**Carlos Mendez**] fue atendido."
            assert pipeline.processed_count == 1
    
    def test_pipeline_run_empty_text(self, mock_strategy_identify, sample_records):
        """Test pipeline run with empty text."""
        pipeline = AnonymizationPipeline(mock_strategy_identify)
        
        # Create record with empty text
        empty_record = {
            "id": "empty_test",
            "text": "",
            "identify": "",
            "masked_text": ""
        }
        
        results = pipeline.run([empty_record])
        
        assert len(results) == 1
        assert results[0]["id"] == "empty_test"
        assert "error" in results[0]
        assert results[0]["error"] == "Empty text"
        assert pipeline.processed_count == 0
    
    def test_pipeline_run_with_max_documents(self, mock_strategy_identify, sample_records):
        """Test pipeline run with max documents limit."""
        with patch.dict(os.environ, {'MAX_DOCUMENTS_TO_PROCESS': '1'}):
            pipeline = AnonymizationPipeline(mock_strategy_identify)
            
            # Mock the identification processor
            with patch.object(pipeline.identification, 'process') as mock_process:
                mock_process.return_value = {
                    "anonymized_text": "El paciente [**Juan García**] fue atendido.",
                    "entities": ["Juan García"]
                }
                
                results = pipeline.run(sample_records)
                
                # Should only process 1 document
                assert len(results) == 1
                assert pipeline.processed_count == 1
    
    def test_pipeline_run_with_first_document(self, mock_strategy_identify, sample_records):
        """Test pipeline run with first document skip."""
        with patch.dict(os.environ, {'FIRST_DOCUMENT_TO_PROCESS': '1'}):
            pipeline = AnonymizationPipeline(mock_strategy_identify)
            
            # Mock the identification processor
            with patch.object(pipeline.identification, 'process') as mock_process:
                mock_process.return_value = {
                    "anonymized_text": "El paciente [**Juan García**] fue atendido.",
                    "entities": ["Juan García"]
                }
                
                results = pipeline.run(sample_records)
                
                # Should skip first document and process second
                assert len(results) == 1
                assert results[0]["id"] == "CARMEN-I_CC_2"
                assert pipeline.processed_count == 1
    
    def test_pipeline_run_with_processing_error(self, mock_strategy_identify, sample_records):
        """Test pipeline run with processing error."""
        pipeline = AnonymizationPipeline(mock_strategy_identify)
        
        # Mock the identification processor to raise an exception
        with patch.object(pipeline.identification, 'process') as mock_process:
            mock_process.side_effect = Exception("Processing error")
            
            results = pipeline.run(sample_records[:1])
            
            assert len(results) == 1
            assert results[0]["id"] == "CARMEN-I_CC_1"
            assert "error" in results[0]
            assert results[0]["error"] == "Processing error"
            assert pipeline.processed_count == 0
    
    def test_identify_method(self, mock_strategy_identify):
        """Test identify method."""
        pipeline = AnonymizationPipeline(mock_strategy_identify)
        
        with patch.object(pipeline.identification, 'process') as mock_process:
            mock_process.return_value = {
                "anonymized_text": "Identified text",
                "entities": ["entity1"]
            }
            
            result = pipeline.identify("Test text")
            
            mock_process.assert_called_once_with("Test text")
            assert result == {
                "anonymized_text": "Identified text",
                "entities": ["entity1"]
            }
    
    def test_anonymize_method_with_anonymizer(self, mock_strategy_label):
        """Test anonymize method with anonymizer."""
        pipeline = AnonymizationPipeline(mock_strategy_label)
        
        with patch.object(pipeline.anonymizer, 'process') as mock_process:
            mock_process.return_value = {
                "anonymized_text": "Anonymized text",
                "entities": ["LABEL"]
            }
            
            identified_result = {"anonymized_text": "Identified text", "entities": ["entity1"]}
            result = pipeline.anonymize("Test text", identified_result)
            
            mock_process.assert_called_once_with("Test text")
            assert result == {
                "anonymized_text": "Anonymized text",
                "entities": ["LABEL"]
            }
    
    def test_anonymize_method_without_anonymizer(self, mock_strategy_identify):
        """Test anonymize method without anonymizer."""
        pipeline = AnonymizationPipeline(mock_strategy_identify)
        
        identified_result = {"anonymized_text": "Identified text", "entities": ["entity1"]}
        result = pipeline.anonymize("Test text", identified_result)
        
        assert result == identified_result
    
    def test_get_max_documents_from_env_valid(self, mock_strategy_identify):
        """Test getting max documents from environment with valid value."""
        with patch.dict(os.environ, {'MAX_DOCUMENTS_TO_PROCESS': '10'}):
            pipeline = AnonymizationPipeline(mock_strategy_identify)
            assert pipeline.max_documents == 10
    
    def test_get_max_documents_from_env_invalid(self, mock_strategy_identify):
        """Test getting max documents from environment with invalid value."""
        with patch.dict(os.environ, {'MAX_DOCUMENTS_TO_PROCESS': 'invalid'}):
            pipeline = AnonymizationPipeline(mock_strategy_identify)
            assert pipeline.max_documents is None
    
    def test_get_max_documents_from_env_zero(self, mock_strategy_identify):
        """Test getting max documents from environment with zero value."""
        with patch.dict(os.environ, {'MAX_DOCUMENTS_TO_PROCESS': '0'}):
            pipeline = AnonymizationPipeline(mock_strategy_identify)
            assert pipeline.max_documents is None
    
    def test_get_max_documents_from_env_not_set(self, mock_strategy_identify):
        """Test getting max documents from environment when not set."""
        with patch.dict(os.environ, {}, clear=True):
            pipeline = AnonymizationPipeline(mock_strategy_identify)
            assert pipeline.max_documents is None