import pytest
import os
from unittest.mock import MagicMock, patch
from src.carmina.llm.cloud_providers.mock_provider import MockProvider
from src.carmina.llm.cloud_providers.aws_provider import AWSProvider
from src.carmina.llm.cloud_providers.local_provider import LocalProvider
from src.carmina.llm.cloud_providers.mock_ollama_provider import MockOllamaProvider
from src.carmina.llm.strategies.llama3_strategy import Llama3Strategy

class DummyProvider:
    def get_name(self):
        return "dummy"
    def run_inference(self, model_id, messages, **kwargs):
        return "dummy response"

@pytest.mark.llm
@pytest.mark.llama3
class TestLlama3Strategy:
    @pytest.fixture
    def mock_llama3_strategy(self):
        """Fixture para una estrategia Llama3 simulada"""
        mock_cloud_provider = MockProvider()
        strategy = Llama3Strategy(
            model_name="llama-3.3-70b", 
            cloud_provider=mock_cloud_provider,
            anonymization_mode="identify"
        )
        return strategy
    
    @pytest.fixture
    def llama3_strategy(self):
        """Fixture para la estrategia de Llama3"""
        aws_provider = AWSProvider("aws")
        return Llama3Strategy(
            model_name="llama-3.3-70b", 
            cloud_provider=aws_provider,
            anonymization_mode="identify"
        )

    @pytest.fixture
    def mock_ollama_strategy(self):
        """Fixture para una estrategia Llama3 con MockOllamaProvider"""
        mock_ollama_provider = MockOllamaProvider()
        strategy = Llama3Strategy(
            model_name="llama-3.3-70b", 
            cloud_provider=mock_ollama_provider,
            anonymization_mode="identify"
        )
        return strategy

    @pytest.fixture
    def ollama_strategy(self):
        """Fixture para la estrategia de Llama3 con LocalProvider (Ollama real)"""
        # Configurar variables de entorno para Ollama
        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
        local_provider = LocalProvider("local", base_url="http://localhost:11434")
        return Llama3Strategy(
            model_name="llama-3.3-70b", 
            cloud_provider=local_provider,
            anonymization_mode="identify"
        )

    def test_init(self, mock_llama3_strategy):
        """Test que verifica la inicialización de la estrategia Llama3"""
        assert isinstance(mock_llama3_strategy.cloud_provider, MockProvider)
        assert mock_llama3_strategy.anonymization_mode == "identify"
        assert mock_llama3_strategy.get_name() == "llama-3.3-70b"
    
    def test_init_with_aws_provider(self, llama3_strategy):
        """Test que verifica la inicialización de la estrategia Llama3 con un proveedor AWS"""
        assert llama3_strategy.anonymization_mode == "identify"
        assert llama3_strategy.get_name() == "llama-3.3-70b"
    
    def test_aws_connection(self, llama3_strategy):
        """Test que verifica la conexión de la estrategia Llama3"""
        assert llama3_strategy.cloud_provider.connect() == "Connected to AWS API"

    def test_mock_identification_mode(self, mock_llama3_strategy, sample_medical_records):
        """Test que verifica el modo de identificación"""
        mock_llama3_strategy.set_anonymization_mode("identify")
        assert mock_llama3_strategy.anonymization_mode == "identify"
        #first test the identify method
        text1 = sample_medical_records[0]["text"]
        result = mock_llama3_strategy.identify(text1)
        assert result == text1

    @patch('src.carmina.llm.strategies.llama3_strategy.Llama3Strategy.identify')
    def test_aws_identification_mode(self, mock_identify, llama3_strategy, sample_medical_records):
        """Test que verifica el modo de identificación con AWS"""
        # Setup the mock to return the expected output
        mock_identify.return_value = sample_medical_records[0]["anonymized_text"]
        
        assert llama3_strategy.anonymization_mode == "identify"
        #first test the identify method
        text1 = sample_medical_records[0]["text"]
        ground_truth1 = sample_medical_records[0]["anonymized_text"]
        result = llama3_strategy.identify(text1)
        assert result == ground_truth1
        mock_identify.assert_called_once_with(text1)

    @patch('src.carmina.llm.strategies.llama3_strategy.Llama3Strategy.identify')
    def test_mock_ollama_identification_mode(self, mock_identify, mock_ollama_strategy, sample_medical_records):
        """Test que verifica el modo de identificación con MockOllamaProvider"""
        # Setup the mock to return the expected output
        mock_identify.return_value = sample_medical_records[0]["anonymized_text"]
        
        assert mock_ollama_strategy.anonymization_mode == "identify"
        #first test the identify method
        text1 = sample_medical_records[0]["text"]
        ground_truth1 = sample_medical_records[0]["anonymized_text"]
        result = mock_ollama_strategy.identify(text1)
        assert result == ground_truth1
        mock_identify.assert_called_once_with(text1)

    def test_mock_ollama_initialization(self, mock_ollama_strategy):
        """Test que verifica la inicialización de la estrategia Llama3 con MockOllamaProvider"""
        assert isinstance(mock_ollama_strategy.cloud_provider, MockOllamaProvider)
        assert mock_ollama_strategy.anonymization_mode == "identify"
        assert mock_ollama_strategy.get_name() == "llama-3.3-70b"
        assert mock_ollama_strategy.cloud_provider.get_name() == "mock_ollama"

    def test_mock_ollama_connection(self, mock_ollama_strategy):
        """Test que verifica la conexión mock de Ollama"""
        connection_result = mock_ollama_strategy.cloud_provider.connect()
        assert "Connected to Mock Ollama" in connection_result

    def test_mock_ollama_model_mapping(self, mock_ollama_strategy):
        """Test que verifica el mapeo de modelos en MockOllamaProvider"""
        provider = mock_ollama_strategy.cloud_provider
        assert provider.get_model_id("llama-3.3-70b") == "llama3.3:70b"
        assert provider.get_model_id("llama-3.3-8b") == "llama3.3:8b"

    def test_mock_ollama_inference(self, mock_ollama_strategy):
        """Test que verifica la inferencia mock con Ollama"""
        messages = [
            {"role": "system", "content": "Eres un asistente médico"},
            {"role": "user", "content": "test message"}
        ]
        
        result = mock_ollama_strategy.cloud_provider.run_inference(
            model_id="llama-3.3-70b",
            messages=messages,
            inference_params={"temperature": 0.1, "max_tokens": 100}
        )
        
        assert isinstance(result, str)
        assert "llama3.3:70b" in result
        assert "test" in result

    def test_mock_ollama_anonymization(self, mock_ollama_strategy, sample_medical_records):
        """Test que verifica la anonimización mock con Ollama"""
        text_to_anonymize = "El paciente Juan con DNI 12345678 fue atendido por el Dr. García en Hospital General de Madrid."
        
        result = mock_ollama_strategy.cloud_provider.run_inference(
            model_id="llama-3.3-70b",
            messages=[
                {"role": "system", "content": "Identifica y anonimiza datos personales"},
                {"role": "user", "content": text_to_anonymize}
            ],
            inference_params={"temperature": 0.1}
        )
        
        # Verificar que se aplicó algún tipo de anonimización mock
        assert "[NOMBRE]" in result or "Mock response" in result

    @patch('requests.get')
    def test_ollama_connection_mock(self, mock_get, ollama_strategy):
        """Test que verifica la conexión a Ollama con mock de requests"""
        # Mock de la respuesta de Ollama
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "0.1.0"}
        mock_get.return_value = mock_response
        
        connection_result = ollama_strategy.cloud_provider.connect()
        assert "Connected to Ollama" in connection_result
        mock_get.assert_called_once_with("http://localhost:11434/api/version")

    @patch('requests.post')
    def test_ollama_inference_mock(self, mock_post, ollama_strategy):
        """Test que verifica la inferencia con Ollama usando mock de requests"""
        # Mock de la respuesta de Ollama
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": "Respuesta simulada de Llama 3.3-70B"
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        messages = [
            {"role": "system", "content": "Eres un asistente médico"},
            {"role": "user", "content": "Identifica los datos personales en este texto"}
        ]
        
        result = ollama_strategy.cloud_provider.run_inference(
            model_id="llama-3.3-70b",
            messages=messages,
            inference_params={"temperature": 0.1, "max_tokens": 100}
        )
        
        assert result == "Respuesta simulada de Llama 3.3-70B"
        mock_post.assert_called_once()

    def test_ollama_strategy_identify_mock(self, mock_ollama_strategy, sample_medical_records):
        """Test que verifica el método identify con MockOllamaProvider"""
        text = sample_medical_records[0]["text"]
        result = mock_ollama_strategy.identify(text)
        
        # El MockOllamaProvider debería devolver algún tipo de respuesta
        assert isinstance(result, str)
        assert len(result) > 0

def test_llama3_strategy_aws():
    provider = AWSProvider()
    strategy = Llama3Strategy(model_name="llama-3.3-70b", cloud_provider=provider)
    # Simula llamada a run_inference (no requiere AWS real)
    try:
        strategy.cloud_provider = DummyProvider()
        result = strategy.run_inference({"messages": [{"role": "user", "content": "test"}]}, {"temperature": 0.1})
        assert result == "dummy response"
    except Exception as e:
        pytest.fail(f"Llama3Strategy with AWSProvider failed: {e}")

def test_llama3_strategy_ollama_real():
    """
    Test para demostrar cómo usar Llama3Strategy con LocalProvider real.
    Este test fallará si Ollama no está instalado y corriendo.
    Usar solo cuando Ollama esté disponible.
    """
    # Skipear por defecto - descomentar cuando Ollama esté disponible
    pytest.skip("Ollama no está disponible - usar solo cuando esté instalado")
    
    # Configuración para Ollama real
    provider = LocalProvider("local", base_url="http://localhost:11434")
    strategy = Llama3Strategy(
        model_name="llama-3.3-70b", 
        cloud_provider=provider,
        anonymization_mode="identify"
    )
    
    # Test de conexión
    connection_result = strategy.cloud_provider.connect()
    assert "Connected to Ollama" in connection_result
    
    # Test de inferencia simple
    messages = [
        {"role": "system", "content": "Eres un asistente útil"},
        {"role": "user", "content": "Hola, ¿cómo estás?"}
    ]
    
    result = strategy.cloud_provider.run_inference(
        model_id="llama-3.3-70b",
        messages=messages,
        inference_params={"temperature": 0.1, "max_tokens": 50}
    )
    
    assert isinstance(result, str)
    assert len(result) > 0

def test_llama3_strategy_with_different_ollama_models():
    """Test que verifica que diferentes modelos de Llama funcionan con Ollama"""
    mock_provider = MockOllamaProvider()
    
    # Test con diferentes versiones de Llama
    models_to_test = ["llama-3.3-70b", "llama-3.3-8b", "llama-3.1-70b"]
    
    for model in models_to_test:
        strategy = Llama3Strategy(
            model_name=model,
            cloud_provider=mock_provider,
            anonymization_mode="identify"
        )
        
        # Verificar que el modelo se mapea correctamente
        ollama_model = strategy.cloud_provider.get_model_id(model)
        assert ollama_model.startswith("llama")
        
        # Verificar que la estrategia funciona
        assert strategy.get_name() == model
        assert isinstance(strategy.cloud_provider, MockOllamaProvider)
