import pytest
from src.carmina.llm.strategies.gemini_strategy import GeminiStrategy
from src.carmina.llm.cloud_providers.google_ai_studio_provider import GoogleAIStudioProvider
from src.carmina.llm.cloud_providers.local_provider import LocalProvider

class DummyProvider:
    def get_name(self):
        return "dummy"
    def run_inference(self, model_id, messages, **kwargs):
        return "dummy response"

def test_gemini_strategy_google():
    provider = GoogleAIStudioProvider(api_key="fake-key")
    strategy = GeminiStrategy(model_name="gemini-2.5-pro", cloud_provider=provider)
    # Simula llamada a run_inference (no requiere API real)
    try:
        strategy.cloud_provider = DummyProvider()
        result = strategy.run_inference([{"role": "user", "content": "test"}], {"temperature": 0.1})
        assert result == "dummy response"
    except Exception as e:
        pytest.fail(f"GeminiStrategy with GoogleAIStudioProvider failed: {e}")

def test_gemma_strategy_local():
    provider = LocalProvider()
    strategy = GeminiStrategy(model_name="gemma-3-1b", cloud_provider=provider)
    # Simula llamada a run_inference (no requiere modelo local real)
    try:
        strategy.cloud_provider = DummyProvider()
        result = strategy.run_inference([{"role": "user", "content": "test"}], {"temperature": 0.1})
        assert result == "dummy response"
    except Exception as e:
        pytest.fail(f"GeminiStrategy with LocalProvider (Gemma) failed: {e}")
