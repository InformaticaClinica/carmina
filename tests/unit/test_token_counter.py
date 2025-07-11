"""
Unit tests for token counting functionality.
"""
import pytest
from src.carmina.llm.utils.token_counter import (
    BaseTokenCounter, OpenAITokenCounter, AnthropicTokenCounter,
    GenericTokenCounter, MockTokenCounter, get_token_counter,
    count_prompt_tokens
)


@pytest.mark.unit
class TestTokenCounters:
    """Test token counting utilities."""
    
    def test_mock_token_counter(self):
        """Test mock token counter."""
        counter = MockTokenCounter()
        
        # Test basic counting
        count = counter.count_tokens("Hello world")
        assert isinstance(count, int)
        assert count > 0
        
        # Test empty string
        assert counter.count_tokens("") == 0
    
    def test_generic_token_counter(self):
        """Test generic token counter."""
        counter = GenericTokenCounter()
        
        # Test basic counting
        count = counter.count_tokens("Hello world test")
        assert isinstance(count, int)
        assert count > 0
        
        # Should count approximately 1.3 tokens per word
        expected = int(3 * 1.3)  # 3 words
        assert abs(count - expected) <= 1
        
        # Test empty string
        assert counter.count_tokens("") == 0
    
    def test_anthropic_token_counter(self):
        """Test Anthropic token counter."""
        counter = AnthropicTokenCounter()
        
        # Test basic counting
        count = counter.count_tokens("Hello world")
        assert isinstance(count, int)
        assert count > 0
        
        # Should be approximately chars/3.7
        expected = int(len("Hello world") / 3.7)
        assert abs(count - expected) <= 1
    
    def test_base_token_counter_message_counting(self):
        """Test base token counter message counting."""
        counter = MockTokenCounter()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello world"}
        ]
        
        counts = counter.count_message_tokens(messages)
        
        assert isinstance(counts, dict)
        assert "system" in counts
        assert "user" in counts
        assert "total" in counts
        assert counts["total"] == counts["system"] + counts["user"]
        assert all(isinstance(v, int) for v in counts.values())
    
    def test_get_token_counter_factory(self):
        """Test token counter factory function."""
        
        # Test OpenAI models
        counter = get_token_counter("gpt-4", "openai")
        assert isinstance(counter, (OpenAITokenCounter, GenericTokenCounter))
        
        # Test Anthropic models
        counter = get_token_counter("claude-3-sonnet", "anthropic")
        assert isinstance(counter, AnthropicTokenCounter)
        
        # Test mock models
        counter = get_token_counter("mock-model", "mock")
        assert isinstance(counter, MockTokenCounter)
        
        # Test unknown models
        counter = get_token_counter("unknown-model", "unknown")
        assert isinstance(counter, GenericTokenCounter)
    
    def test_count_prompt_tokens_function(self):
        """Test the utility function for counting prompt tokens."""
        
        system_prompt = "You are a helpful assistant."
        user_prompt = "Please help me with this task."
        
        counts = count_prompt_tokens(system_prompt, user_prompt, "mock-model", "mock")
        
        assert isinstance(counts, dict)
        assert "system" in counts
        assert "user" in counts
        assert "total" in counts
        assert counts["total"] == counts["system"] + counts["user"]
        assert all(v >= 0 for v in counts.values())


@pytest.mark.unit  
class TestTokenCounterIntegration:
    """Test token counter integration with strategies."""
    
    def test_openai_token_counter_availability(self):
        """Test that OpenAI token counter works when tiktoken is available."""
        try:
            import tiktoken
            counter = OpenAITokenCounter("gpt-4")
            
            # Test that it can count tokens
            count = counter.count_tokens("Hello world")
            assert isinstance(count, int)
            assert count > 0
            
        except ImportError:
            # If tiktoken is not available, test should pass
            with pytest.raises(ImportError):
                OpenAITokenCounter("gpt-4")
    
    def test_token_counting_with_special_characters(self):
        """Test token counting with special characters and multiple languages."""
        counter = GenericTokenCounter()
        
        # Test with special characters
        text_with_special = "Hello, world! ¿Cómo estás? 你好世界"
        count = counter.count_tokens(text_with_special)
        assert isinstance(count, int)
        assert count > 0
        
        # Test with empty and whitespace
        assert counter.count_tokens("") == 0
        assert counter.count_tokens("   ") == 0
        assert counter.count_tokens("\n\t") == 0
    
    def test_token_counter_consistency(self):
        """Test that token counters are consistent for same input."""
        counter = MockTokenCounter()
        
        text = "This is a consistent test text for token counting."
        
        # Count multiple times - should be consistent
        count1 = counter.count_tokens(text)
        count2 = counter.count_tokens(text)
        count3 = counter.count_tokens(text)
        
        assert count1 == count2 == count3