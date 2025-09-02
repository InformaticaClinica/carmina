#!/usr/bin/env python3

import os
import sys
sys.path.append('.')

# Set environment variables for quick test
os.environ["FIRST_DOCUMENT_TO_PROCESS"] = "101" 
os.environ["MAX_DOCUMENTS_TO_PROCESS"] = "1"

from src.carmina.tools.benchmark_runner import BenchmarkRunner
from src.carmina.utils.logging_config import setup_logging

print("🧪 Running quick validation test with updated alignment...")

logger = setup_logging()
logger.info("Starting quick validation")

try:
    BenchmarkRunner().run()
    logger.info("Quick validation completed successfully")
    print("✅ Test completed successfully!")
except Exception as e:
    print(f"❌ Error during test: {e}")
    raise