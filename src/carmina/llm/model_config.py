"""Configuration parameters for supported LLM models."""

MODEL_CONFIGS = {
    # OpenAI models
    "gpt-3.5-turbo": {
        "context_window": 16385,
        "strategy": "openai",
        "capabilities": ["identification", "anonymization"]
    },
    "gpt-4-turbo": {
        "context_window": 128000,
        "strategy": "openai",
        "capabilities": ["identification", "anonymization"]
    },
    
    # Anthropic models
    "claude-3-sonnet": {
        "context_window": 180000,
        "strategy": "anthropic",
        "capabilities": ["identification", "anonymization"]
    },
    "claude-3-haiku": {
        "context_window": 150000,
        "strategy": "anthropic",
        "capabilities": ["identification", "anonymization"]  
    },
    "claude-3.5-sonnet": {
        "context_window": 200000,
        "strategy": "anthropic",
        "capabilities": ["identification", "anonymization"]
    },
    "claude-3.7-sonnet": {
        "context_window": 220000,
        "strategy": "anthropic",
        "capabilities": ["identification", "anonymization"]
    },
    
    # Qwen 3.5 models
    "qwen-3.5-27b": {
        "context_window": 131072,
        "strategy": "qwen",
        "capabilities": ["identification", "anonymization"]
    },
    "qwen-3.5-122b": {
        "context_window": 131072,
        "strategy": "qwen",
        "capabilities": ["identification", "anonymization"]
    },
    # HuggingFace models
    "llama-3-8b": {
        "context_window": 8192,
        "strategy": "huggingface",
        "capabilities": ["identification", "anonymization"]
    },
    "llama-3-70b": {
        "context_window": 8192,
        "strategy": "huggingface",
        "capabilities": ["identification", "anonymization"]
    },
    "mistral-7b": {
        "context_window": 8192,
        "strategy": "huggingface",
        "capabilities": ["identification", "anonymization"]
    }
}