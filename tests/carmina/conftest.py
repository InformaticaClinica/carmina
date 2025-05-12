import pytest
import os
from unittest.mock import MagicMock

@pytest.fixture
def cloud_credentials():
    """Fixture providing cloud provider credentials from environment"""
    return {
        "aws": {
            "access_key": os.environ.get("AWS_ACCESS_KEY_ID"),
            "secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY")
        },
        "azure": {
            "api_key": os.environ.get("AZURE_API_KEY"),
            "endpoint": os.environ.get("AZURE_ENDPOINT")
        },
        "openai": {
            "api_key": os.environ.get("OPENAI_API_KEY")
        }
    }

@pytest.fixture
def sample_medical_texts():
    """Fixture providing sample medical texts for testing"""
    return [
        "El paciente Juan García fue atendido por el Dr. Martínez el 12/05/2023.",
        "María López (DNI 12345678A) presenta dolor abdominal desde hace 3 días.",
        "Historia clínica de Pedro Sánchez, nacido el 07/01/1975, teléfono 612345678."
    ]

@pytest.fixture
def sample_medical_records():
    """Fixture que proporciona registros médicos de ejemplo con ID y texto"""
    return [
        {
            "id": "doc1", 
            "text": "El paciente Juan García fue atendido.",
            "anonymized_text": "El paciente [**Juan García**] fue atendido.",
            "entities_identified": ["Juan García"],
            },
    ]

@pytest.fixture
def mock_llm_strategy():
    """Fixture providing a mocked LLM strategy"""
    strategy = MagicMock()
    strategy.anonymization_mode = "identify"
    strategy.get_name.return_value = "MockStrategy"
    return strategy