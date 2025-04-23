"""Prompt management package."""

import os
from typing import Dict, List

# Map of available prompts by type
AVAILABLE_PROMPTS: Dict[str, List[str]] = {
    "anonymize": ["label", "substitute"],
    "identify": []
}

# Base directory for prompts
PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))

def list_available_prompts():
    """List all available prompts in this package."""
    result = {}
    for prompt_type, modes in AVAILABLE_PROMPTS.items():
        if modes:
            for mode in modes:
                result[f"{prompt_type}_{mode}"] = os.path.join(
                    PROMPTS_DIR, f"{prompt_type}_{mode}.txt"
                )
        else:
            result[prompt_type] = os.path.join(
                PROMPTS_DIR, f"{prompt_type}.txt"
            )
    return result