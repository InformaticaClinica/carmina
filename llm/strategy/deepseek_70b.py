import json
from .llm_strategy import LLMStrategy
from llm.prompt_handling.ollama_prompt_handler import OllamaPromptHandler
import requests

class Deepseek_70b(LLMStrategy):
    def __init__(self):
        super().__init__()
        self._model_id = "deepseek-r1:70b"
        self.OLLAMA_URL ="http://host.docker.internal:11434/api/generate"
        self._prompt_handler = OllamaPromptHandler()

    def create_body(self, prompt: str) -> str:
        prompt["model"] = self._model_id
        prompt["stream"] = False
        return json.dumps(prompt)
    
    def invoke_model(self, data: str) -> str:
        print("before invoke")
        response = requests.post(self.OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
        print(response)
        return response.json()["response"]


    def generate_prompt(self, model_prompt: dict) -> str:
        prompt = self._prompt_handler.format_prompt(model_prompt)
        body = self.create_body(prompt)
        return self.invoke_model(body)
