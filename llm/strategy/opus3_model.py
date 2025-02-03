import boto3
import json
from .llm_strategy import LLMStrategy
from llm.prompt_handling.haiku_prompt_handler import HaikuPromptHandler
from botocore.config import Config
config = Config(connect_timeout=9000, read_timeout=9000, retries={'max_attempts': 3})

class OpusModel(LLMStrategy):
    def __init__(self):
        super().__init__()
        self._client = boto3.client('bedrock-runtime', region_name='us-west-2', config=config)
        self._model_id = "anthropic.claude-3-opus-20240229-v1:0"
        self._prompt_handler = HaikuPromptHandler()
        self._top_k = 50


    def create_body(self, prompt, system_prompt) -> str:
        return json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "messages": prompt,
            "max_tokens":self._max_gen_len,
            "temperature":self._temperature,
            "top_p":self._top_p,
            "top_k":self._top_k
        })

    def invoke_model(self, body: dict) -> str:
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=body,
        )
        response = json.loads(response.get('body').read())
        return response["content"][0]["text"]


    def generate_prompt(self, model_prompt: dict) -> str:
        [system_prompt, prompt] = self._prompt_handler.format_prompt(model_prompt)
        body = self.create_body(prompt, system_prompt)
        return self.invoke_model(body)