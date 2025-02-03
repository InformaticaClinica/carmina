import json
import os
from openai import OpenAI
from .llm_strategy import LLMStrategy
from llm.prompt_handling.chatgpt_prompt_handler import ChatGPTPromptHandler

class ChatGPTModel(LLMStrategy):
    def __init__(self):
        super().__init__()
        #self._apikey = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-" #enter api key
        self._client = OpenAI()
        self._prompt_handler = ChatGPTPromptHandler()

    def invoke_model(self, model_ver="gpt-4o", numbers=1, temperature_setting=1.0, messages=[]):
        #print(messages)
        response = self._client.chat.completions.create(
            model=model_ver,
            n=numbers,
            temperature=temperature_setting,
            messages=messages
        )    
        return response.choices[0].message.content

    def generate_prompt(self, system_content="", replaced=""):
        prompt = self._prompt_handler.format_prompt(system_content, replaced)
        return self.invoke_model(model_ver="gpt-4o", numbers=1, temperature_setting=1.0, messages=prompt) #or either enter gpt-4o-mini in model_ver
