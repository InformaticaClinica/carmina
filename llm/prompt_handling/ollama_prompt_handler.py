from .prompt_handler import PromptHandler

class OllamaPromptHandler(PromptHandler):
    def format_prompt(self, data: dict):
        data2 = {}
        data2["system"] = data["system"]
        data2["prompt"] = data["user"]
        return data2