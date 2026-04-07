"""Utility functions for loading and formatting prompt templates."""

import os
from typing import Optional

def load_system_prompt(prompt_type: str, anonymization_mode: Optional[str] = None) -> str:
    """
    Load a prompt from the appropriate XML file as plain text.
    
    Args:
        prompt_type: The type of prompt (identify, anonymize) or full path like "system/identify", "user/label"
        anonymization_mode: Mode of anonymization if applicable (label, substitute)
        
    Returns:
        The prompt as a string
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if prompt_type contains a path (like "system/identify" or "user/label")
    if "/" in prompt_type:
        # Split the path and use it directly
        path_parts = prompt_type.split("/")
        prompts_dir = os.path.join(base_dir, "prompts", path_parts[0])
        prompt_name = path_parts[1]
        
        # Build the filename based on parameters
        if prompt_name == "anonymize" and anonymization_mode:
            filename = f"{prompt_name}_{anonymization_mode}.xml"
        else:
            filename = f"{prompt_name}.xml"
    else:
        # Default behavior: use system directory
        prompts_dir = os.path.join(base_dir, "prompts", "system")
        
        # Build the filename based on parameters
        if prompt_type == "anonymize" and anonymization_mode:
            filename = f"{prompt_type}_{anonymization_mode}.xml"
        else:
            filename = f"{prompt_type}.xml"
    
    file_path = os.path.join(prompts_dir, filename)
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()
