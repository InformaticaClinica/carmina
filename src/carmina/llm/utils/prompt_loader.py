"""Utility functions for loading and formatting prompt templates."""

import os
from typing import Optional

def load_system_prompt(prompt_type: str, anonymization_mode: Optional[str] = None) -> str:
    """
    Load a system prompt from the appropriate file.
    
    Args:
        prompt_type: The type of prompt (identify, anonymize)
        anonymization_mode: Mode of anonymization if applicable (label, substitute)
        
    Returns:
        The system prompt as a string
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_dir = os.path.join(base_dir, "prompts")
    
    # Build the filename based on parameters
    if prompt_type == "anonymize" and anonymization_mode:
        filename = f"{prompt_type}_{anonymization_mode}.txt"
    else:
        filename = f"{prompt_type}.txt"
    
    file_path = os.path.join(prompts_dir, filename)
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()