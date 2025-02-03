from .prompt_handler import PromptHandler

class MistralPromptHandler(PromptHandler):
    def format_prompt(self, data: dict):
        return f"""<s>[INST]{data["system"]}

            {data["user"]}[/INST]"""