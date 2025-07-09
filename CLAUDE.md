# CLAUDE.md - Carmina Healthcare Text Anonymization System

## Project Overview

**Carmina** is a sophisticated healthcare text anonymization system designed to process clinical records and anonymize personally identifiable information (PII) according to European regulations. The system leverages multiple Large Language Models (LLMs) to perform intelligent text anonymization while maintaining semantic meaning for clinical research purposes.

### Core Purpose
- **Healthcare Data Anonymization**: Anonymize clinical texts while preserving medical relevance
- **Multi-LLM Benchmarking**: Compare performance across different LLM providers
- **Regulatory Compliance**: Align with European healthcare data protection regulations
- **Research Enablement**: Generate anonymized datasets for clinical research

## Project Structure

```
hc-carmina/
├── src/carmina/                    # Main application code
│   ├── config.py                   # Global configuration management
│   ├── model_executor.py           # Model execution orchestration
│   ├── data_sources/               # Data loading and extraction
│   │   ├── data_loader.py          # Main data loading interface
│   │   ├── loaders/                # File format specific loaders
│   │   └── extractors/             # Entity extraction utilities
│   ├── llm/                        # LLM integration framework
│   │   ├── base.py                 # Core LLM interfaces
│   │   ├── factory.py              # LLM instance creation
│   │   ├── strategies/             # LLM-specific implementations
│   │   ├── cloud_providers/        # Cloud service integrations
│   │   ├── prompts/                # Prompt templates and management
│   │   └── utils/                  # LLM utilities
│   ├── pipeline/                   # Processing pipeline
│   │   ├── anon_pipeline.py        # Main anonymization pipeline
│   │   └── processors/             # Stage-specific processors
│   ├── metrics/                    # Evaluation and metrics
│   │   ├── evaluator.py            # Performance evaluation
│   │   ├── classification.py       # Classification metrics
│   │   └── similarity.py           # Text similarity metrics
│   ├── tools/                      # Utility tools
│   │   ├── benchmark_runner.py     # Benchmarking orchestration
│   │   └── benchmark_summary.py    # Results summarization
│   └── utils/                      # General utilities
├── data/                           # Data directory
│   ├── inputs/                     # Input data (CARMEN corpus)
│   └── outputs/                    # Generated outputs and metrics
├── tests/                          # Test suites
├── docs/                           # Documentation
└── main.py                         # Application entry point
```

## Core Architecture

### 1. Strategy Pattern Implementation

The system implements a sophisticated strategy pattern for LLM integration:

**Base Strategy Interface** (`src/carmina/llm/strategies/base_strategy.py`):
```python
class BaseLLMStrategy(ABC):
    def __init__(self, model_name: str, cloud_provider: BaseCloudProvider, **kwargs):
        self.model_name = model_name
        self.cloud_provider = cloud_provider
        self.anonymization_mode = kwargs.get("anonymization_mode", "identify")
        self.temperature = float(kwargs.get("temperature", 1.0))
        self.max_tokens = int(kwargs.get("max_tokens", 2500))
```

**Supported LLM Providers**:
- **OpenAI**: GPT-4 family models
- **Anthropic**: Claude family models  
- **Google**: Gemini models
- **DeepSeek**: DeepSeek models
- **Meta**: Llama models
- **Mistral**: Mistral models

### 2. Factory Pattern for LLM Creation

**LLM Factory** (`src/carmina/llm/factory.py`):
```python
class LLMFactory:
    @classmethod
    def create(cls, model_name: str, cloud_provider: str, 
               strategy_kwargs: Optional[Dict] = None,
               provider_kwargs: Optional[Dict] = None) -> BaseLLMStrategy:
        provider = CloudProviderFactory.create(cloud_provider, **provider_kwargs)
        strategy_class = cls._get_strategy_for_model(model_name)
        return strategy_class(model_name=model_name, cloud_provider=provider, **strategy_kwargs)
```

### 3. Multi-Cloud Provider Support

**Cloud Providers**:
- **AWS Bedrock**: Enterprise-grade deployment
- **Azure OpenAI**: Microsoft cloud integration
- **Google AI Studio**: Google cloud services
- **OpenAI API**: Direct OpenAI integration
- **Local**: Local model deployment
- **Mock**: Testing and development

### 4. Anonymization Pipeline

**Three-Stage Process** (`src/carmina/pipeline/anon_pipeline.py`):

1. **Identification Stage**: Detect PII entities
2. **Anonymization Stage**: Apply anonymization strategy
3. **Validation Stage**: Ensure quality and compliance

**Pipeline Configuration**:
```python
pipeline = AnonymizationPipeline(llm_strategy)
anonymized_records = pipeline.run(records)
```

## Anonymization Modes

### 1. IDENTIFY Mode
**Purpose**: Detect and mark personal identifiers
**Output**: Original text with PII enclosed in `[** **]` markers

**Example**:
```
Input:  "El paciente Juan García fue atendido por el Dr. Martínez"
Output: "El paciente [**Juan García**] fue atendido por el Dr. [**Martínez**]"
```

**Detected Entities**:
- Personal names (patients, healthcare staff)
- Ages and demographic information
- Dates and timestamps
- Phone numbers and identifiers
- Institutions and locations
- URLs and sensitive information

### 2. LABEL Mode
**Purpose**: Replace PII with semantic labels
**Output**: Text with standardized category labels

**Example**:
```
Input:  "Paciente de [**41 años**], [**albañil**]"
Output: "Paciente de [**EDAD_SUJETO_ASISTENCIA**], [**PROFESION**]"
```

**Standard Labels**:
- `EDAD_SUJETO_ASISTENCIA` (patient age)
- `PROFESION` (profession)
- `FECHAS` (dates)
- `NUMERO_TELEFONO` (phone numbers)
- `INSTITUCION` (institutions)
- `NOMBRE_PERSONAL_SANITARIO` (healthcare staff names)
- `SEXO_SUJETO_ASISTENCIA` (gender)
- `TERRITORIO` (geographical locations)

### 3. SUBSTITUTE Mode
**Purpose**: Replace PII with realistic alternatives
**Output**: Text with plausible substitute values

**Example**:
```
Input:  "La paciente [**María López**], de [**32 años**], acudió el [**15/03/2022**]"
Output: "La paciente [**Ana Torres**], de [**45 años**], acudió el [**14/03/2025**]"
```

## Data Processing

### Input Data Format (CARMEN Corpus)

**Expected Structure**:
```
data/inputs/
├── CARMEN1_mappings.tsv    # Language and metadata mapping
└── txt/
    ├── ann/                # Annotation files (.ann)
    ├── raw/                # Raw text files (.txt)
    ├── identify/           # Generated: identified entities
    └── masked/             # Generated: masked versions
```

**Data Loader** (`src/carmina/data_sources/data_loader.py`):
```python
def load_dataset(input_path: str) -> List[Dict[str, Any]]:
    # Supports: JSON, CSV, TXT files and directory structures
    # Returns: List of records with 'text' and 'labels' fields
```

### Output Data Format

**Generated Files**:
- **Anonymized Texts**: Individual `.txt` files with anonymized content
- **Metrics**: Performance evaluation in JSON format
- **Debug Files**: Detailed processing logs and intermediate results

## Evaluation and Metrics

### Performance Metrics (`src/carmina/metrics/evaluator.py`)

**Classification Metrics**:
- **Precision**: Correctly identified entities / Total identified entities
- **Recall**: Correctly identified entities / Total actual entities
- **F1-Score**: Harmonic mean of precision and recall

**Similarity Metrics**:
- **Cosine Similarity**: Semantic similarity between texts
- **Levenshtein Distance**: Character-level edit distance
- **Inverse Levenshtein**: Normalized similarity score

**Evaluation Functions**:
```python
def evaluate_identification(ground_truth_records, prediction_records):
    # Returns identification performance metrics
    
def evaluate_label(ground_truth_texts, prediction_texts, 
                   ground_truth_labels, prediction_labels):
    # Returns labeling quality metrics
```

## Configuration Management

### Environment Variables (`src/carmina/config.py`)

**Core Configuration**:
```bash
# Model Configuration
DEFAULT_MODEL=claude-3.7-sonnet
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=1000

# Processing Configuration
ANONYMIZATION_MODE=label
CLOUD_PROVIDER=aws
MODELS=claude-3.7-sonnet,gpt-4-turbo

# Data Paths
INPUT_DIR=data/input.json
OUTPUT_DIR=data/outputs/
METRICS_DIR=metrics/
DEBUG_DIR=data/outputs/debug/

# Processing Limits
MAX_DOCUMENTS_TO_PROCESS=100
FIRST_DOCUMENT_TO_PROCESS=0

# Logging
DEBUG_ENABLED=False
LOG_LEVEL=INFO
```

### Model Configuration (`src/carmina/llm/model_config.py`)

**Model Specifications**:
```python
MODEL_CONFIGS = {
    "claude-3.5-sonnet": {
        "context_window": 200000,
        "strategy": "anthropic",
        "capabilities": ["identification", "anonymization"]
    },
    "gpt-4-turbo": {
        "context_window": 128000,
        "strategy": "openai",
        "capabilities": ["identification", "anonymization"]
    }
}
```

## Development Commands

### Running the Application

**Main Application**:
```bash
python main.py
```

**With Docker**:
```bash
docker-compose up -d
docker-compose down
```

### Testing

**All Tests**:
```bash
pytest
```

**Specific Test Categories**:
```bash
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/e2e/           # End-to-end tests
```

### Environment Setup

**Python Environment**:
```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

**Environment Variables**:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Benchmarking System

### Benchmark Runner (`src/carmina/tools/benchmark_runner.py`)

**Multi-Model Execution**:
```python
class BenchmarkRunner:
    def run(self):
        for model in self.models:
            executor = ModelExecutor(
                model_name=model,
                anonymization_mode=self.anonymization_mode,
                cloud_provider=self.cloud_provider,
                input_path=self.input_path,
                output_dir=self.output_dir,
                metrics_dir=self.metrics_dir
            )
            executor.execute()
```

### Results Analysis (`src/carmina/tools/benchmark_summary.py`)

**Performance Comparison**:
- Model-by-model performance metrics
- Aggregated benchmark results
- Statistical analysis and rankings

## Prompt Engineering

### Prompt Templates (`src/carmina/llm/prompts/`)

**System Prompts** (`system/`):
- `identify.xml`: Entity identification instructions
- `label.xml`: Label classification guidelines
- `substitute.xml`: Substitution generation rules

**User Prompts** (`user/`):
- Task-specific prompt templates
- Input text placeholder management
- Context-aware prompt generation

### Prompt Loading (`src/carmina/llm/utils/prompt_loader.py`)

**Dynamic Prompt Management**:
```python
def load_system_prompt(prompt_type: str, anonymization_mode: str) -> str:
    # Loads XML prompts based on mode and type
    # Supports both system and user prompts
```

## Error Handling and Logging

### Logging Configuration (`src/carmina/utils/logging_config.py`)

**Structured Logging**:
- Application-wide logging configuration
- Level-based message filtering
- File and console output support

### Exception Handling (`src/carmina/data_sources/exceptions.py`)

**Custom Exceptions**:
```python
class DataLoadError(Exception):
    """Raised when data loading fails"""
    pass
```

## Integration Guidelines

### Adding New LLM Providers

1. **Create Strategy Class**: Implement `BaseLLMStrategy`
2. **Register in Factory**: Add to `LLMFactory._strategies`
3. **Configure Provider**: Add cloud provider if needed
4. **Add Model Config**: Define model specifications

### Extending Anonymization Modes

1. **Create Processor**: Implement `BaseProcessor`
2. **Add Prompt Templates**: Create system/user prompts
3. **Update Pipeline**: Register in `AnonymizationPipeline`
4. **Add Evaluation**: Implement metrics calculation

## Performance Optimization

### Processing Limits

**Environment Controls**:
- `MAX_DOCUMENTS_TO_PROCESS`: Limit batch size
- `FIRST_DOCUMENT_TO_PROCESS`: Skip initial documents
- Rate limiting and timeout management

### Caching and Optimization

**Efficiency Features**:
- Prompt template caching
- Token counting optimization
- Batch processing support
- Parallel execution capabilities

## Security Considerations

### Data Protection

**Privacy Measures**:
- No logging of sensitive data
- Secure API key management
- Encrypted data transmission
- Compliance with healthcare regulations

### Model Security

**Safe Deployment**:
- Input validation and sanitization
- Output filtering and validation
- Rate limiting and abuse prevention
- Audit logging and monitoring

## Troubleshooting

### Common Issues

**Configuration Problems**:
- Check `.env` file for API keys
- Verify cloud provider credentials
- Ensure model availability

**Processing Errors**:
- Check input data format
- Verify file permissions
- Monitor rate limits

**Performance Issues**:
- Adjust batch sizes
- Optimize prompt templates
- Monitor token usage

## Contributing

### Development Setup

1. Clone repository
2. Create virtual environment
3. Install dependencies
4. Configure environment variables
5. Run tests to verify setup

### Code Standards

- Follow existing code patterns
- Add comprehensive tests
- Document new features
- Maintain backward compatibility

---

*This documentation provides a comprehensive guide to the Carmina healthcare text anonymization system. For specific implementation details, refer to the source code and inline documentation.*