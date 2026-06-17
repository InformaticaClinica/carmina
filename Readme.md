# Carmina

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![GitHub last commit](https://img.shields.io/github/last-commit/InformaticaClinica/carmina)](https://github.com/InformaticaClinica/carmina/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/InformaticaClinica/carmina)](https://github.com/InformaticaClinica/carmina/issues)
[![GitHub forks](https://img.shields.io/github/forks/InformaticaClinica/carmina)](https://github.com/InformaticaClinica/carmina/network)
[![GitHub stars](https://img.shields.io/github/stars/InformaticaClinica/carmina)](https://github.com/InformaticaClinica/carmina/stargazers)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/InformaticaClinica/carmina/actions)
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->
Carmina is an application designed for clinical text anonymization and performance comparison between different methods and models. It uses large language models (LLMs) to process and analyze text data, focusing on meddocan guidelines for text anonymization according to European regulations.

## Features
* Support for multiple LLM providers:
  * **OpenAI** (GPT models)
  * **Anthropic** (Claude models via AWS Bedrock)
  * **Google AI Studio** (Gemini Developer API)
  * **Google Vertex AI** (Gemini on Google Cloud Platform) - [Setup Guide](docs/vertex_ai_setup.md)
  * **Azure OpenAI** (DeepSeek, GPT models)
  * **Local** (Ollama - Llama, Gemma, Qwen, Mistral)
* Configurable data processing pipeline
* Evaluation of model performance metrics
* Benchmarking tools to compare different LLMs
* Compatible with different cloud providers

## Prerequisites
The project is designed to be executed primarily using Docker Compose, which simplifies dependency management. Alternatively, it can be run in a Python environment. The necessary libraries are listed in `requirements.txt`.

* Python 3.11 or higher (if not using Docker)
* Docker and Docker Compose (recommended)
* API keys for the LLM services you wish to use (configured in the `.env` file)

## Installation

The recommended way to install and run Carmina is through Docker Compose.

### Option 1: Using Docker Compose (Recommended)
1. Clone the repository:
    ```bash
    git clone https://github.com/InformaticaClinica/carmina
    cd carmina
    ```
2. Copy the example environment variables file:
    ```bash
    cp .env.example .env
    ```
3. Edit the `.env` file and add your API keys and other necessary configurations:
    ```env
    # Example content for .env
    OPENAI_API_KEY="your_openai_api_key"
    ANTHROPIC_API_KEY="your_anthropic_api_key"
    # Add other necessary configuration variables
    ```
    
    > **Note**: Currently, only one cloud provider can be used at a time. For example, if you configure Azure as a provider, you can only use the models available on that platform.

4. Start the services with Docker Compose:
    ```bash
    docker-compose up -d
    ```

### Option 2: Using Python Virtual Environment
1. Clone the repository:
    ```bash
    git clone https://github.com/InformaticaClinica/carmina
    cd carmina
    ```
2. Copy the example environment variables file:
    ```bash
    cp .env.example .env
    ```
3. Edit the `.env` file and add your API keys and other necessary configurations.
4. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
5. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### With Docker Compose
Once the services are up with `docker-compose up -d`, the application (currently the `BenchmarkRunner`) will run automatically according to the configuration in `docker-compose.yaml`.

To stop the services:
```bash
docker-compose down
```

### With Python Virtual Environment
If you've set up a virtual environment, you can run the main application (which currently runs the `BenchmarkRunner`) with:
```bash
python main.py
```
This will start the benchmarking process configured in `src/carmina/tools/benchmark_runner.py`. The results and logs will be saved in the `data/outputs/` and `logs/` directories respectively.

### Data Preparation (CARMEN Corpus)

Carmina expects a local `data/` directory. The repository includes only the empty folder skeleton and ignores real datasets, generated outputs, logs, metrics, credentials, and zipped data files.

```
data/
├── inputs/
│   ├── CARMEN1_mappings.tsv
│   └── txt/
│       ├── ann/            # .ann annotation files
│       ├── raw/            # original .txt files
│       ├── masked/         # masked .txt files with entity placeholders
│       └── identify/       # identified spans wrapped with [**...**]
└── outputs/
    ├── debug/
    │   └── logs/
    ├── logs/
    └── metrics/
```

Place your `.ann` files in `data/inputs/txt/ann/` and the corresponding raw `.txt` files in `data/inputs/txt/raw/`. The `CARMEN1_mappings.tsv` file must be present in `data/inputs/`. For full details, see [docs/data.md](docs/data.md).

## Documentation

The documentation folder is committed as normal Markdown and PlantUML files:

* [Documentation index](docs/README.md)
* [Data directory structure](docs/data.md)
* [Output files and metrics](docs/output.md)
* [Testing guide](docs/tests.md)
* [AWS rate limiting](docs/aws_rate_limiting.md)
* [Vertex AI setup](docs/vertex_ai_setup.md)

## Testing
To verify that the models and application logic work correctly, you can run unit and integration tests using `pytest`. Make sure you have the development dependencies installed (included in `requirements.txt` or available in the Docker development image if defined).


From the project root (with the virtual environment activated or inside the Docker container if configured for development):
```bash
pytest
```
You can find more details about the tests in `docs/tests.md`.

## Project Structure

```
carmina/
├── data/                   # Input and output data
│   ├── inputs/
│   └── outputs/
├── docs/                   # Project documentation
├── logs/                   # Application log files
├── src/                    # Application source code
│   └── carmina/
│       ├── config.py           # Global configurations
│       ├── model_executor.py   # Logic for executing LLM models
│       ├── data_sources/       # Modules for loading and extracting data
│       ├── llm/                # Abstractions and clients for LLMs
│       ├── metrics/            # Modules for calculating metrics
│       ├── pipeline/           # Processing pipeline logic
│       ├── tools/              # Additional tools (e.g., benchmarking)
│       └── utils/              # General utilities
├── tests/                  # Unit, integration, and e2e tests
├── .env.example            # Template for environment variables
├── .gitignore
├── CHANGELOG.md
├── docker-compose.yaml     # Docker Compose configuration
├── Dockerfile              # Docker image definition
├── main.py                 # Main application entry point
├── pytest.ini              # Pytest configuration
├── Readme.md               # This file
└── requirements.txt        # Python dependencies
```

## Changelog Summary

### Version [1.1.0] - 2026-04-20
* Support for additional model/provider strategies (including remote Ollama API integration).
* Expanded prompt and anonymization behavior for identify/label/substitute flows.
* Pipeline robustness improvements (retry/cleanup and output backup handling).
* Repository cleanup for release branches (removed `AGENTS.md` and `test_vertex_quick.py`).

### Version [1.0.0] - 2025-05-20
* Initial project setup with `main.py` entry point.
* Docker support with `Dockerfile` and `docker-compose.yaml`.
* Data loading and processing capabilities under `src/carmina/data_sources/`.
* LLM integration framework in `src/carmina/llm/`, supporting multiple cloud providers and custom prompts.
* Metrics calculation and evaluation system in `src/carmina/metrics/`.
* Data processing pipeline implemented in `src/carmina/pipeline/`, including anonymization features.
* Benchmarking tools for performance analysis in `src/carmina/tools/`.
* Comprehensive testing suite under `tests/` covering unit, integration, and end-to-end tests.
* Project documentation including flow diagrams and output specifications in `docs/`.
* Configuration management via `src/carmina/config.py`.
* Utility functions for file operations and logging in `src/carmina/utils/`.

For a full list of changes, please refer to the [CHANGELOG.md](CHANGELOG.md) file.

## Contributions

Contributions are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/new-feature`).
3. Make your changes and commit (`git commit -am 'Add new feature'`).
4. Push your changes to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

Please make sure your changes pass the tests and follow the project style guidelines.

> **Note**: The testing system is currently under development. Some tests may not pass correctly while we work on their complete implementation.

## License

To be defined

## Contact

Petter Peñafiel / Hospital Clínic de Barcelona (Clinical Informatics Service) - papenafiel@clinic.cat
