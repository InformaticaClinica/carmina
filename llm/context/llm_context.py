from llm.strategy.llm_strategy import LLMStrategy

class LLMContext:
    def __init__(self, strategy: LLMStrategy):
        self._strategy = strategy

    def create_client(self):
        return self._strategy.create_client()

    def generate_response(self, prompt: dict) -> str:
        return self._strategy.generate_prompt(prompt)