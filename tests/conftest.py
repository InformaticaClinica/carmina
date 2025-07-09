"""
Global pytest configuration and fixtures for Carmina tests.
"""
import pytest
import os
import json
from unittest.mock import MagicMock, Mock
from typing import Dict, List, Any


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Fixture to set up test environment variables."""
    # Store original values
    original_values = {}
    test_env_vars = {
        "ANONYMIZATION_MODE": "identify",
        "CLOUD_PROVIDER": "mock",
        "DEFAULT_MODEL": "mock-model",
        "DEFAULT_TEMPERATURE": "0.1",
        "DEFAULT_MAX_TOKENS": "1000",
        "OUTPUT_DIR": "tests/output",
        "METRICS_DIR": "tests/metrics",
        "DEBUG_DIR": "tests/debug",
        "DEBUG_ENABLED": "True",
        "LOG_LEVEL": "INFO"
    }
    
    # Store original values and set test values
    for key, value in test_env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield
    
    # Restore original values
    for key, original_value in original_values.items():
        if original_value is not None:
            os.environ[key] = original_value
        elif key in os.environ:
            del os.environ[key]


@pytest.fixture
def sample_medical_texts():
    """Sample medical texts in Spanish for testing."""
    return [
        "El paciente Juan García fue atendido por el Dr. Martínez el 12/05/2023.",
        "María López (DNI 12345678A) presenta dolor abdominal desde hace 3 días.",
        "Historia clínica de Pedro Sánchez, nacido el 07/01/1975, teléfono 612345678.",
        "Paciente de 45 años, carpintero, con antecedentes de diabetes.",
        "Consulta en Hospital Clínic de Barcelona el 15/03/2024."
    ]


@pytest.fixture
def sample_identified_texts():
    """Sample identified texts with PII marked."""
    return [
        "El paciente [**Juan García**] fue atendido por el Dr. [**Martínez**] el [**12/05/2023**].",
        "[**María López**] (DNI [**12345678A**]) presenta dolor abdominal desde hace [**3 días**].",
        "Historia clínica de [**Pedro Sánchez**], nacido el [**07/01/1975**], teléfono [**612345678**].",
        "Paciente de [**45 años**], [**carpintero**], con antecedentes de diabetes.",
        "Consulta en [**Hospital Clínic de Barcelona**] el [**15/03/2024**]."
    ]


@pytest.fixture
def sample_labeled_texts():
    """Sample labeled texts with semantic labels."""
    return [
        "El paciente [**OTROS_SUJETO_ASISTENCIA**] fue atendido por el Dr. [**NOMBRE_PERSONAL_SANITARIO**] el [**FECHAS**].",
        "[**OTROS_SUJETO_ASISTENCIA**] (DNI [**NUMERO_IDENTIF**]) presenta dolor abdominal desde hace [**FECHAS**].",
        "Historia clínica de [**OTROS_SUJETO_ASISTENCIA**], nacido el [**FECHAS**], teléfono [**NUMERO_TELEFONO**].",
        "Paciente de [**EDAD_SUJETO_ASISTENCIA**], [**PROFESION**], con antecedentes de diabetes.",
        "Consulta en [**HOSPITAL**] el [**FECHAS**]."
    ]


@pytest.fixture
def sample_records():
    """Sample medical records with all processing stages."""
    return [
        {
            "id": "CARMEN-I_CC_1",
            "text": "El paciente Juan García fue atendido por el Dr. Martínez.",
            "identify": "El paciente [**Juan García**] fue atendido por el Dr. [**Martínez**].",
            "masked_text": "El paciente [**OTROS_SUJETO_ASISTENCIA**] fue atendido por el Dr. [**NOMBRE_PERSONAL_SANITARIO**].",
            "identified_text": "",
            "anonymized_text": "",
            "entities_identified": [],
            "entities_anonymized": []
        },
        {
            "id": "CARMEN-I_CC_2",
            "text": "María López presenta dolor abdominal desde hace 3 días.",
            "identify": "[**María López**] presenta dolor abdominal desde hace [**3 días**].",
            "masked_text": "[**OTROS_SUJETO_ASISTENCIA**] presenta dolor abdominal desde hace [**FECHAS**].",
            "identified_text": "",
            "anonymized_text": "",
            "entities_identified": [],
            "entities_anonymized": []
        }
    ]


@pytest.fixture
def mock_cloud_provider():
    """Mock cloud provider for testing."""
    provider = MagicMock()
    provider.get_name.return_value = "mock_provider"
    provider.run_inference.return_value = "Mock response from LLM"
    return provider


@pytest.fixture
def mock_llm_strategy():
    """Mock LLM strategy for testing."""
    strategy = MagicMock()
    strategy.model_name = "mock-model"
    strategy.anonymization_mode = "identify"
    strategy.get_name.return_value = "MockStrategy"
    strategy.get_context_window.return_value = 4096
    strategy.count_tokens.return_value = 100
    return strategy


@pytest.fixture
def mock_identification_processor():
    """Mock identification processor."""
    processor = MagicMock()
    processor.process.return_value = {
        "anonymized_text": "El paciente [**Juan García**] fue atendido.",
        "entities": ["Juan García"]
    }
    return processor


@pytest.fixture
def mock_labeling_processor():
    """Mock labeling processor."""
    processor = MagicMock()
    processor.process.return_value = {
        "anonymized_text": "El paciente [**OTROS_SUJETO_ASISTENCIA**] fue atendido.",
        "entities": ["OTROS_SUJETO_ASISTENCIA"]
    }
    return processor


@pytest.fixture
def mock_substitution_processor():
    """Mock substitution processor."""
    processor = MagicMock()
    processor.process.return_value = {
        "anonymized_text": "El paciente [**Carlos Mendez**] fue atendido.",
        "entities": ["Carlos Mendez"]
    }
    return processor


@pytest.fixture
def temp_test_files(tmp_path):
    """Create temporary test files."""
    # Create test directories
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # Create sample input files
    sample_data = [
        {"id": "test1", "text": "Test text 1"},
        {"id": "test2", "text": "Test text 2"}
    ]
    
    json_file = input_dir / "test_data.json"
    with open(json_file, 'w') as f:
        json.dump(sample_data, f)
    
    txt_file = input_dir / "test_data.txt"
    with open(txt_file, 'w') as f:
        f.write("Sample text content")
    
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "json_file": str(json_file),
        "txt_file": str(txt_file)
    }


@pytest.fixture
def sample_metrics():
    """Sample metrics for testing."""
    return {
        "identification_precision": 0.85,
        "identification_recall": 0.90,
        "identification_f1": 0.875,
        "label_precision": 0.80,
        "label_recall": 0.85,
        "label_f1": 0.825,
        "label_cosine_sim": 0.95,
        "label_levenshtein": 5.2,
        "label_inv_levenshtein": 0.80,
        "label_overall": 4.25
    }


@pytest.fixture
def mock_metrics_recorder():
    """Mock metrics recorder for testing."""
    recorder = MagicMock()
    recorder.model_name = "test-model"
    recorder.metrics = {}
    recorder.record_metric = MagicMock()
    recorder.record_all = MagicMock()
    recorder.export_to_json = MagicMock()
    return recorder


# Markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.llm = pytest.mark.llm
pytest.mark.slow = pytest.mark.slow