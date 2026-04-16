# AGENTS.md - Guidelines for Agentic Coding

## Build/Test Commands
- `make setup` - Create venv and install dependencies  
- `make test` - Run all tests
- `make run` - Run main application
- `python -m pytest tests/unit/test_file.py -v` - Run single test
- `python -m pytest -m unit` - Run unit tests only
- `python -m pytest -m integration` - Run integration tests only

## Code Style Guidelines
- **Imports**: Use absolute imports from `src.carmina.*`
- **Formatting**: Standard Python formatting, 4-space indentation
- **Types**: Use type hints for all function signatures and class attributes
- **Naming**: 
  - Classes: PascalCase (e.g., `BaseLLMStrategy`)
  - Functions/variables: snake_case
  - Constants: UPPER_SNAKE_CASE
- **Error Handling**: Use try/except blocks, raise custom exceptions from `src.carmina.data_sources.exceptions`
- **Abstract Classes**: Inherit from ABC, use `@abstractmethod` decorators
- **Testing**: Use pytest fixtures from `conftest.py`, mock external dependencies
- **Environment**: Use environment variables for configuration, see `conftest.py` for test defaults