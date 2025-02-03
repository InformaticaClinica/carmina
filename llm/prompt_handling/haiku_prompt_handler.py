from .prompt_handler import PromptHandler

class HaikuPromptHandler(PromptHandler):
    def format_prompt(self, data: dict):
        return [data["system"], [
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "text",
                        "text": f"""{data["user"]}"""
                    }
                    ]
                }
            ]]