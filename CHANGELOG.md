# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-05-20

### Added
- Initial project setup with `main.py` entry point.
- Docker support with `Dockerfile` and `docker-compose.yaml`.
- Data loading and processing capabilities under `src/carmina/data_sources/`.
- LLM integration framework in `src/carmina/llm/`, supporting multiple cloud providers and custom prompts.
- Metrics calculation and evaluation system in `src/carmina/metrics/`.
- Data processing pipeline implemented in `src/carmina/pipeline/`, including anonymization features.
- Benchmarking tools for performance analysis in `src/carmina/tools/`.
- Comprehensive testing suite under `tests/` covering unit, integration, and end-to-end tests.
- Project documentation including flow diagrams and output specifications in `docs/`.
- Configuration management via `src/carmina/config.py`.
- Utility functions for file operations and logging in `src/carmina/utils/`.