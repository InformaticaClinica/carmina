# Testing Guide for Carmina Anonymization Pipeline
## Table of Contents
1. Introduction
2. Testing Structure
3. Setting Up Your Test Environment
4. Writing Tests
5. Running Tests
6. CI/CD Integration
7. Best Practices

## Introduction
This guide explains our testing approach for the Carmina anonymization pipeline. We follow a comprehensive testing strategy that includes unit tests, integration tests, model-specific tests, and end-to-end tests.

Our approach is designed to:
- Ensure individual components work correctly in isolation
- Verify that our integrations with cloud providers function properly
- Validate that LLM models perform as expected
- Confirm that the entire pipeline delivers the correct outputs

## Testing Structure

```text
tests/
├── conftest.py
├── e2e/
│   └── test_complete_workflow.py
├── integration/
│   ├── test_anonymization_pipeline.py
│   ├── test_benchmark_runner.py
│   ├── test_llm_integration.py
│   └── test_vertex_ai_integration.py
└── unit/
    ├── test_base_strategy.py
    ├── test_config.py
    ├── test_data_loader.py
    ├── test_llm_factory.py
    ├── test_metrics.py
    ├── test_metrics_recorder.py
    ├── test_model_executor.py
    ├── test_token_counter.py
    ├── test_vertex_ai_provider.py
    ├── test_vertex_ai_provider_enhancements.py
    └── test_vertex_gemini_strategy.py
```


Test Types Explained
1. Unit Tests: Focus on testing individual components in isolation by mocking dependencies.
2. Integration Tests: Verify that our code correctly interacts with external cloud provider APIs.
3. LLM Tests: Test specific LLM models to ensure they perform correctly for our use case.
4. End-to-End Tests: Test the complete anonymization workflow from input to output.

## Setting Up Your Test Environment
### Prerequisites

1. Make sure you have pytest installed:
```bash
pip install pytest pytest-cov
```

2. Configure environment variables for API access:
```.env
# For AWS tests
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# For Azure tests
AZURE_API_KEY=your_api_key
AZURE_ENDPOINT=your_endpoint

# For OpenAI tests
OPENAI_API_KEY=your_api_key
```

3. Creating a conftest.py
The conftest.py file contains shared fixtures that can be used across multiple test files:

```python
import pytest
import os
from unittest.mock import MagicMock

@pytest.fixture
def cloud_credentials():
    """Fixture providing cloud provider credentials from environment"""
    return {
        "aws": {
            "access_key": os.environ.get("AWS_ACCESS_KEY_ID"),
            "secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY")
        },
        "azure": {
            "api_key": os.environ.get("AZURE_API_KEY"),
            "endpoint": os.environ.get("AZURE_ENDPOINT")
        },
        "openai": {
            "api_key": os.environ.get("OPENAI_API_KEY")
        }
    }
```

## Writing Tests
1. Unit Tests
Unit tests focus on testing a single component in isolation.

2. Integration Tests
Integration tests verify that our code works correctly with cloud provider APIs.

3. LLM Tests
LLM tests focus on the specific capabilities of each language model.

4. End-to-End Tests
End-to-end tests verify that the complete workflow functions correctly.

## Running Tests
### Running All Tests
To run all tests:
```
pytest
```

### Running Specific Test Types
To run only unit tests:
```
pytest tests/unit/
```

To run only integration tests:
```
pytest tests/integration/
```

To run only end-to-end tests:
```
pytest tests/e2e/
```
### Running Tests with Specific Markers
To run only AWS integration tests:
```
pytest -m aws_integration
```
To run only end-to-end tests:
```
pytest -m e2e
```
Running Tests with Coverage
To run tests and generate a coverage report:
```
pytest --cov=src/carmina
```
To generate an HTML coverage report:
```
pytest --cov=src/carmina --cov-report=html
```
## CI/CD Integration
Add this section to your .github/workflows/tests.yml file:
```
IN PROCESS...
```

## Best Practices
1. Focus on Behavior: Test what a component does, not how it's implemented.
2. Use Meaningful Names: Test names should describe what they're testing.
3. Follow AAA Pattern: Arrange (set up), Act (execute), Assert (verify).
4. Mock External Dependencies: Use mocks for API calls in unit tests.
5. Use Fixtures: Create reusable test fixtures in conftest.py.
6. Test Edge Cases: Include tests for empty inputs, invalid inputs, etc.
7. Group Related Tests: Use TestClass for grouping related tests.
8. Skip Tests When Needed: Use @pytest.mark.skipif for tests requiring credentials or specific environments.
9. Mark Test Categories: Use @pytest.mark for categorizing tests (e.g., @pytest.mark.e2e).
10. Keep Tests Independent: Each test should be able to run independently of others.

Remember: The goal of testing is to ensure that your code works correctly, not just to achieve high code coverage!
