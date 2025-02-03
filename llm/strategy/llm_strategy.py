from abc import ABC, abstractmethod

class LLMStrategy(ABC):
    def __init__(self):
        self._temperature = 0.1
        self._max_gen_len = 2048
        self._top_p = 0.9

    @abstractmethod
    def generate_prompt(self, prompt: dict) -> str:
        pass
    
    def set_temperature(self, temperature: float):
        if 0 <= temperature <= 1:
            self.temperature = temperature
        else:
            raise ValueError("Temperature must be between 0 and 1.")

    def set_max_gen_len(self, max_gen_len: int):
        if max_gen_len > 0:
            self.max_gen_len = max_gen_len
        else:
            raise ValueError("Maximum generation length must be greater than 0.")

    def set_top_p(self, top_p: float):
        if 0 <= top_p <= 1:
            self.top_p = top_p
        else:
            raise ValueError("Top_p value must be between 0 and 1.")