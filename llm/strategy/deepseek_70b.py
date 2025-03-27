import json

from llm.prompt_handling.chatgpt_prompt_handler import ChatGPTPromptHandler
from .llm_strategy import LLMStrategy
from llm.prompt_handling.ollama_prompt_handler import OllamaPromptHandler
import os

import requests
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

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


class Deepseek_70b_azure(LLMStrategy):
    def __init__(self):
        super().__init__()
        self._model_id = "deepseek-r1:70b_azure"
        self.DEPLOYMENT_URL = os.getenv("AZURE_DEEPSEEK_ENDPOINT")
        self.API_KEY = os.getenv("AZURE_DEEPSEEK_KEY")
        if self.DEPLOYMENT_URL is None:
            raise ValueError("AZURE_DEEPSEEK_ENDPOINT environment variable not set.")
        if self.API_KEY is None:
            raise ValueError("AZURE_DEEPSEEK_KEY environment variable not set.")
        self._client = ChatCompletionsClient(
            endpoint=self.DEPLOYMENT_URL,
            credential=AzureKeyCredential(self.API_KEY)
        )
        self._prompt_handler = ChatGPTPromptHandler()

    def invoke_model(self, messages, temperature=0.6, max_tokens=32768):
        if max_tokens > 32768:
            raise ValueError("Maximum tokens must be less than or equal to 32768.")
        response = self._client.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )    
        return response.choices[0].message.content

    def generate_prompt(self, model_prompt: dict) -> str:
        prompt = self._prompt_handler.format_prompt(model_prompt)
        response = self.invoke_model(prompt)
        # we parse out the reasoning in the <think> tags
        idx = response.find("</think>")
        if idx != -1:
            # deepseek adds a few newlines so we remove them as well
            response = response[idx + 8:].lstrip()
        return response