from .prompt_handler import PromptHandler

class Llama3PromptHandler(PromptHandler):
    def format_prompt(self, data: dict) -> str:
        prompt = f"""
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            {data["system"]}
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {data["user"]}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
        """
        return prompt