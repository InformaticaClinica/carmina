"""
Integration tests for LLM strategy integrations.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.carmina.llm.factory import LLMFactory
from src.carmina.llm.strategies.mock_strategy import MockLLMStrategy
from src.carmina.llm.cloud_providers.mock_provider import MockCloudProvider


@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests for LLM components."""
    
    @pytest.fixture
    def mock_provider(self):
        """Mock cloud provider."""
        return MockCloudProvider()
    
    def test_llm_factory_creates_mock_strategy(self):
        """Test LLM factory creates mock strategy correctly."""
        llm = LLMFactory.create(
            model_name="mock-model",
            cloud_provider="mock",
            strategy_kwargs={"anonymization_mode": "identify"}
        )
        
        assert isinstance(llm, MockLLMStrategy)
        assert llm.model_name == "mock-model"
        assert llm.anonymization_mode == "identify"
    
    def test_mock_strategy_identify_workflow(self, mock_provider):
        """Test mock strategy identify workflow."""
        strategy = MockLLMStrategy(
            model_name="mock-model",
            cloud_provider=mock_provider,
            anonymization_mode="identify"
        )
        
        text = "El paciente Juan García fue atendido."
        result = strategy.identify(text)
        
        assert isinstance(result, str)
        assert "Juan García" in result or "[**" in result
    
    def test_mock_strategy_label_workflow(self, mock_provider):
        """Test mock strategy label workflow."""
        strategy = MockLLMStrategy(
            model_name="mock-model",
            cloud_provider=mock_provider,
            anonymization_mode="label"
        )
        
        text = "El paciente [**Juan García**] fue atendido."
        result = strategy.label(text)
        
        assert isinstance(result, str)
        assert "OTROS_SUJETO_ASISTENCIA" in result or "[**" in result
    
    def test_mock_strategy_substitute_workflow(self, mock_provider):
        """Test mock strategy substitute workflow."""
        strategy = MockLLMStrategy(
            model_name="mock-model",
            cloud_provider=mock_provider,
            anonymization_mode="substitute"
        )
        
        text = "El paciente [**Juan García**] fue atendido."
        result = strategy.substitute(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_mock_strategy_process_for_anonymization_identify(self, mock_provider):
        """Test mock strategy process_for_anonymization in identify mode."""
        strategy = MockLLMStrategy(
            model_name="mock-model",
            cloud_provider=mock_provider,
            anonymization_mode="identify"
        )
        
        text = "El paciente Juan García fue atendido."
        result = strategy.process_for_anonymization(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_mock_strategy_process_for_anonymization_label(self, mock_provider):
        """Test mock strategy process_for_anonymization in label mode."""
        strategy = MockLLMStrategy(
            model_name="mock-model",
            cloud_provider=mock_provider,
            anonymization_mode="label"
        )
        
        text = "El paciente [**Juan García**] fue atendido."
        result = strategy.process_for_anonymization(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_mock_strategy_process_for_anonymization_substitute(self, mock_provider):
        """Test mock strategy process_for_anonymization in substitute mode."""
        strategy = MockLLMStrategy(
            model_name="mock-model",
            cloud_provider=mock_provider,
            anonymization_mode="substitute"
        )
        
        text = "El paciente [**Juan García**] fue atendido."
        result = strategy.process_for_anonymization(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_mock_strategy_get_context_window(self, mock_provider):
        """Test mock strategy get_context_window method."""
        strategy = MockLLMStrategy(
            model_name="mock-model",
            cloud_provider=mock_provider
        )
        
        context_window = strategy.get_context_window()
        
        assert isinstance(context_window, int)
        assert context_window > 0
    
    def test_mock_strategy_count_tokens(self, mock_provider):
        """Test mock strategy count_tokens method."""
        strategy = MockLLMStrategy(
            model_name="mock-model",
            cloud_provider=mock_provider
        )
        
        token_count = strategy.count_tokens("This is a test text.")
        
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_mock_strategy_get_name(self, mock_provider):
        """Test mock strategy get_name method."""
        strategy = MockLLMStrategy(
            model_name="mock-model",
            cloud_provider=mock_provider
        )
        
        name = strategy.get_name()
        
        assert isinstance(name, str)
        assert "mock-model" in name
        assert "MockLLMStrategy" in name
    
    @patch('src.carmina.llm.utils.prompt_loader.load_system_prompt')
    def test_mock_strategy_get_message(self, mock_load_prompt, mock_provider):
        """Test mock strategy get_message method."""
        mock_load_prompt.side_effect = [
            "System prompt",
            "User prompt with <input_text>"
        ]
        
        strategy = MockLLMStrategy(
            model_name="mock-model",
            cloud_provider=mock_provider
        )
        
        messages = strategy.get_message("identify", "Test text")
        
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Test text" in messages[1]["content"]
    
    def test_mock_provider_run_inference(self, mock_provider):
        """Test mock provider run_inference method."""
        messages = [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Process this text."}
        ]
        
        result = mock_provider.run_inference("mock-model", messages)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_mock_provider_get_name(self, mock_provider):
        """Test mock provider get_name method."""
        name = mock_provider.get_name()
        
        assert isinstance(name, str)
        assert name == "mock"
    
    def test_llm_factory_with_different_providers(self):
        """Test LLM factory with different cloud providers."""
        providers = ["mock", "local"]
        
        for provider in providers:
            llm = LLMFactory.create(
                model_name="mock-model",
                cloud_provider=provider,
                strategy_kwargs={"anonymization_mode": "identify"}
            )
            
            assert llm is not None
            assert llm.model_name == "mock-model"
            assert llm.anonymization_mode == "identify"
    
    def test_llm_factory_with_different_modes(self):
        """Test LLM factory with different anonymization modes."""
        modes = ["identify", "label", "substitute"]
        
        for mode in modes:
            llm = LLMFactory.create(
                model_name="mock-model",
                cloud_provider="mock",
                strategy_kwargs={"anonymization_mode": mode}
            )
            
            assert llm is not None
            assert llm.anonymization_mode == mode
    
    def test_strategy_with_custom_parameters(self, mock_provider):
        """Test strategy with custom parameters."""
        strategy = MockLLMStrategy(
            model_name="mock-model",
            cloud_provider=mock_provider,
            anonymization_mode="identify",
            temperature=0.5,
            max_tokens=1000,
            top_k=20,
            top_p=0.8
        )
        
        assert strategy.temperature == 0.5
        assert strategy.max_tokens == 1000
        assert strategy.top_k == 20
        assert strategy.top_p == 0.8
    
    def test_end_to_end_anonymization_flow(self):
        """Test end-to-end anonymization flow."""
        # Create LLM instance
        llm = LLMFactory.create(
            model_name="mock-model",
            cloud_provider="mock",
            strategy_kwargs={"anonymization_mode": "identify"}
        )
        
        # Test text processing
        original_text = "El paciente Juan García fue atendido por el Dr. Martínez."
        
        # Step 1: Identify
        identified_text = llm.identify(original_text)
        assert isinstance(identified_text, str)
        assert len(identified_text) > 0
        
        # Step 2: Switch to label mode and process
        llm.anonymization_mode = "label"
        labeled_text = llm.label(identified_text)
        assert isinstance(labeled_text, str)
        assert len(labeled_text) > 0
        
        # Step 3: Switch to substitute mode and process
        llm.anonymization_mode = "substitute"
        substituted_text = llm.substitute(identified_text)
        assert isinstance(substituted_text, str)
        assert len(substituted_text) > 0