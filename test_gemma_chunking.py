#!/usr/bin/env python3
"""
Test script to process a single file with gemma model and debug chunking behavior.
"""

import os
import json
import logging
import dotenv
from src.carmina.llm.factory import LLMFactory
from src.carmina.pipeline.processors.identification_processor import IdentificationProcessor
from src.carmina.metrics.compare_line import extract_all_metrics

# Load environment variables
dotenv.load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_file():
    """Create a sample text file with entities for testing."""
    # Create a much longer text to test chunking
    sample_text = """
    John Doe visited the hospital on January 15, 2023. His phone number is 555-123-4567.
    He lives at 123 Main Street, New York, NY 10001. His email is john.doe@example.com.
    The doctor, Dr. Sarah Johnson, prescribed medication for his condition.
    """ * 20  # Repeat to make it longer

    with open("sample_test.txt", "w", encoding="utf-8") as f:
        f.write(sample_text.strip())

    return "sample_test.txt"

def test_gemma_chunking():
    """Test processing a single file with gemma model."""

    # Create sample file
    sample_file = create_sample_file()
    logger.info(f"Created sample file: {sample_file}")

    # Read the sample text
    with open(sample_file, "r", encoding="utf-8") as f:
        text = f.read()

    logger.info(f"Sample text length: {len(text)} characters")

    # Setup gemma model
    llm = LLMFactory.create(
        model_name="gemma3:1b",
        cloud_provider="local",
        strategy_kwargs={"anonymization_mode": "label"}
    )

    # Debug: Check messages that would be sent
    messages = llm.get_message("identify", text)
    logger.info("Messages being sent to model:")
    for msg in messages:
        logger.info(f"Role: {msg['role']}")
        logger.info(f"Content length: {len(msg['content'])}")
        logger.info(f"Content preview: {msg['content'][:200]}...")

    # Create identification processor
    processor = IdentificationProcessor(llm)

    # Process the text
    logger.info("Processing text with gemma model...")
    result = processor.process(text, filename=sample_file)

    logger.info("Processing result:")
    logger.info(f"Anonymized text: {result.get('anonymized_text', '')}")
    logger.info(f"Entities found: {result.get('entities', {})}")

    # Check chunking details
    # We need to access the chunking method, but it's private. Let's modify to make it accessible for testing
    chunks = processor._chunk_text(text, 100)
    logger.info(f"Text was chunked into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        token_count = llm.count_tokens(chunk)
        logger.info(f"Chunk {i}: {len(chunk)} chars, ~{token_count} tokens")
        logger.info(f"Content: {chunk[:100]}{'...' if len(chunk) > 100 else ''}")

    # Test metrics calculation
    # Create mock data for metrics
    mock_record = {
        "id": "sample_test.txt",
        "text": text,
        "masked_text": text,  # Assuming no masking for this test
        "identify": result.get("anonymized_text", ""),
        "identified_text": result.get("anonymized_text", ""),
        "anonymized_text": result.get("anonymized_text", ""),
        "labels": result.get("entities", {})
    }

    logger.info("Testing metrics calculation...")
    try:
        metrics = extract_all_metrics(
            ground_truth_identity_texts=[mock_record.get("identify", "")],
            prediction_identity_texts=[mock_record["identified_text"]],
            ground_truth_texts=[mock_record["masked_text"]],
            prediction_texts=[mock_record["anonymized_text"]],
            filenames=[mock_record["id"]],
            languages=["en"]
        )
        logger.info(f"Metrics result: {json.dumps(metrics, indent=2)}")
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")

    # Cleanup
    if os.path.exists(sample_file):
        os.remove(sample_file)
        logger.info(f"Cleaned up sample file: {sample_file}")

if __name__ == "__main__":
    test_gemma_chunking()