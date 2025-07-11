#!/usr/bin/env python3
"""
Example of how to use token counting and cost tracking in Carmina.

This example demonstrates:
1. How to count tokens for individual texts
2. How to count tokens for complete prompts (system + user)
3. How to track token usage across different anonymization modes
4. How to record token metrics for cost analysis
"""

import sys
import os

# Add src to path so we can import carmina modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from carmina.llm.factory import LLMFactory
from carmina.metrics.recorder import MetricsRecorder
from carmina.llm.utils.token_counter import count_prompt_tokens


def demonstrate_basic_token_counting():
    """Demonstrate basic token counting with different strategies."""
    print("=== Basic Token Counting ===")
    
    # Create different LLM strategies
    strategies = {
        "mock": LLMFactory.create("mock-model", "mock"),
        "openai": LLMFactory.create("gpt-4", "openai") if hasattr(LLMFactory, 'create') else None,
        "anthropic": LLMFactory.create("claude-3-sonnet", "anthropic") if hasattr(LLMFactory, 'create') else None
    }
    
    test_text = "El paciente Juan García fue atendido por el Dr. Martínez el 15 de marzo de 2024."
    
    for name, strategy in strategies.items():
        if strategy is None:
            continue
            
        print(f"\n{name.upper()} Strategy:")
        print(f"  Text: {test_text}")
        
        try:
            token_count = strategy.count_tokens(test_text)
            print(f"  Token count: {token_count}")
        except Exception as e:
            print(f"  Error counting tokens: {e}")


def demonstrate_prompt_token_counting():
    """Demonstrate counting tokens for complete prompts."""
    print("\n=== Prompt Token Counting ===")
    
    # Create a mock strategy for demonstration
    strategy = LLMFactory.create("mock-model", "mock")
    
    test_text = "El paciente María López fue atendida en el hospital."
    
    # Test different anonymization modes
    modes = ["identify", "label", "substitute"]
    
    for mode in modes:
        print(f"\n{mode.upper()} Mode:")
        try:
            token_counts = strategy.count_prompt_tokens(mode, test_text)
            print(f"  System tokens: {token_counts['system']}")
            print(f"  User tokens: {token_counts['user']}")
            print(f"  Total tokens: {token_counts['total']}")
        except Exception as e:
            print(f"  Error: {e}")


def demonstrate_cost_tracking():
    """Demonstrate cost tracking with token metrics."""
    print("\n=== Cost Tracking ===")
    
    # Model pricing (example rates in USD per 1K tokens)
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "mock-model": {"input": 0.001, "output": 0.002}  # For demo
    }
    
    strategy = LLMFactory.create("mock-model", "mock")
    recorder = MetricsRecorder("mock-model")
    
    test_text = "Paciente de 45 años con diabetes tipo 2."
    
    # Count tokens for anonymization
    token_counts = strategy.count_prompt_tokens("identify", test_text)
    
    # Calculate costs (assuming some output tokens)
    input_tokens = token_counts["total"]
    output_tokens = strategy.count_tokens("El paciente [**EDAD_SUJETO_ASISTENCIA**] con [**CONDICION_MEDICA**].")
    
    model_pricing = pricing["mock-model"]
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    
    # Record metrics
    recorder.record_token_metrics(token_counts, "identify")
    recorder.record_cost_metrics(input_cost, output_cost)
    recorder.record("input_tokens", input_tokens)
    recorder.record("output_tokens", output_tokens)
    recorder.record("filename", "example_document.txt")
    
    # Save metrics
    recorder.save_current()
    
    # Display results
    results = recorder.get_all_results()
    if results:
        result = results[0]
        print(f"  Document: {result['filename']}")
        print(f"  Input tokens: {result['input_tokens']}")
        print(f"  Output tokens: {result['output_tokens']}")
        print(f"  Total tokens: {result['identify_tokens_total']}")
        print(f"  Input cost: ${result['cost_input']:.6f}")
        print(f"  Output cost: ${result['cost_output']:.6f}")
        print(f"  Total cost: ${result['cost_total']:.6f}")


def demonstrate_batch_processing():
    """Demonstrate token tracking for multiple documents."""
    print("\n=== Batch Processing ===")
    
    strategy = LLMFactory.create("mock-model", "mock")
    recorder = MetricsRecorder("mock-model")
    
    # Simulate multiple documents
    documents = [
        "El paciente Juan García fue diagnosticado con hipertensión.",
        "María Rodríguez presenta síntomas de gripe estacional.",
        "El Dr. López realizó una cirugía exitosa al paciente.",
        "Paciente de 67 años con antecedentes de diabetes."
    ]
    
    total_tokens = 0
    total_cost = 0.0
    
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}:")
        print(f"  Text: {doc}")
        
        # Count tokens for identification
        token_counts = strategy.count_prompt_tokens("identify", doc)
        
        # Simulate processing cost
        input_cost = (token_counts["total"] / 1000) * 0.001
        output_cost = (token_counts["total"] / 1000) * 0.002  # Assume similar output size
        
        total_tokens += token_counts["total"]
        total_cost += input_cost + output_cost
        
        # Record metrics for this document
        recorder.record_token_metrics(token_counts, "identify")
        recorder.record_cost_metrics(input_cost, output_cost)
        recorder.record("filename", f"document_{i}.txt")
        recorder.save_current()
        
        print(f"  Tokens: {token_counts['total']}")
        print(f"  Cost: ${input_cost + output_cost:.6f}")
    
    print(f"\nBatch Summary:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total cost: ${total_cost:.6f}")
    print(f"  Average tokens per document: {total_tokens / len(documents):.1f}")
    print(f"  Average cost per document: ${total_cost / len(documents):.6f}")


def demonstrate_utility_functions():
    """Demonstrate standalone utility functions."""
    print("\n=== Utility Functions ===")
    
    system_prompt = "Eres un asistente especializado en anonimización de textos médicos."
    user_prompt = "Por favor identifica y marca la información personal en el siguiente texto: El paciente José Martín fue atendido."
    
    # Count tokens using utility function
    token_counts = count_prompt_tokens(system_prompt, user_prompt, "mock-model", "mock")
    
    print(f"System prompt tokens: {token_counts['system']}")
    print(f"User prompt tokens: {token_counts['user']}")
    print(f"Total prompt tokens: {token_counts['total']}")


if __name__ == "__main__":
    print("Carmina Token Counting and Cost Tracking Examples")
    print("=" * 50)
    
    try:
        demonstrate_basic_token_counting()
        demonstrate_prompt_token_counting()
        demonstrate_cost_tracking()
        demonstrate_batch_processing()
        demonstrate_utility_functions()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)