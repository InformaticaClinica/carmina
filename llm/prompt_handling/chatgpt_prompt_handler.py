from .prompt_handler import PromptHandler

class ChatGPTPromptHandler(PromptHandler):
    def format_prompt(self, system_content="", replaced=""):
        return [
                {"role": "system", "content": system_content['system']},
                {"role": "user", "content": [
                    {"type": "text", "text": system_content['user']},
                ]}
            ]
