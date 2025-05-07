import pytest
from unittest.mock import MagicMock, patch
from src.carmina.pipeline.anon_pipeline import AnonymizationPipeline

class TestIdentificationPipeline:
    """Tests para el pipeline en modo identificación"""
    
    @pytest.fixture
    def mock_identify_strategy(self):
        """Fixture que proporciona una estrategia LLM en modo identification"""
        strategy = MagicMock()
        strategy.anonymization_mode = "identify"
        strategy.get_name.return_value = "MockIdentifyStrategy"
        return strategy
    
    @pytest.fixture
    def expected_entities(self):
        """Fixture que proporciona las entidades que esperamos identificar"""
        return [
            {"type": "NOMBRE", "text": "Juan García", "start": 12, "end": 23},
            {"type": "FECHA", "text": "12/05/2023", "start": 48, "end": 58}
        ]
    
    def test_identification_pipeline_init(self, mock_identify_strategy, sample_medical_texts):
        """Test que el pipeline se inicializa correctamente en modo identify"""
        # Arrange & Act
        pipeline = AnonymizationPipeline(mock_identify_strategy)
        
        # Assert
        assert pipeline.anonymization_mode == "identify"
        assert pipeline.identification is not None
        assert pipeline.anonymizer is None
    
    def test_identification_pipeline_flow(self, mock_identify_strategy, sample_medical_records, expected_entities):
        # Arrange
        pipeline = AnonymizationPipeline(mock_identify_strategy)
        
        # Configurar el procesador de identificación
        pipeline.identification.process = MagicMock()
        print(pipeline.identification.process.return_value)
        
        # Act
        results = pipeline.run([sample_medical_records[0]])  # Usar el primer registro
        
        # Assert
        # ... resto del código