"""
Main entry point for the Carmina application.
"""

from src.carmina.tools.benchmark_runner import BenchmarkRunner
from src.carmina.utils.logging_config import setup_logging

# Entry point for the application
if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting Carmina application")
    
    BenchmarkRunner().run()
    
    logger.info("Carmina application completed successfully")
