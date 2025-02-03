from abc import ABC, abstractmethod

class PromptHandler(ABC):
    @abstractmethod
    def format_prompt(self, data: dict) -> str:
        pass