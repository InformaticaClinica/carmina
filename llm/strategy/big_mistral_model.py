import boto3
import json
from .llm_strategy import LLMStrategy
from botocore.config import Config
config = Config(connect_timeout=9000, read_timeout=9000, retries={'max_attempts': 3})
from llm.prompt_handling.mistral_prompt_handler import MistralPromptHandler

class BigMistralModel(LLMStrategy):
    def __init__(self):
        super().__init__()
        self._model_id = "mistral.mixtral-8x7b-instruct-v0:1"
        self._client = boto3.client('bedrock-runtime', region_name='eu-west-2', config=config)
        self._prompt_handler = MistralPromptHandler()
        self._top_k = 50

    def create_body(self, prompt) -> str:
        return json.dumps({
            "prompt": prompt,
            "max_tokens":self._max_gen_len,
            "temperature":self._temperature,
            "top_p":self._top_p,
            "top_k":self._top_k
        })

    def invoke_model(self, body: dict) -> str:
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=body
        )
        response = json.loads(response.get('body').read())
        return response["outputs"][0]["text"]

    def generate_prompt(self, model_prompt: dict) -> str:
        prompt = self._prompt_handler.format_prompt(model_prompt)
        body = self.create_body(prompt)
        return self.invoke_model(body)