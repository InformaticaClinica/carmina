# Makefile for Carmina Healthcare Text Anonymization System

# Python interpreter
PYTHON := python3.12

# Virtual environment
VENV := venv
VENV_BIN := $(VENV)/bin

# Default target
.DEFAULT_GOAL := help

# Help command
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make venv     - Create virtual environment only"
	@echo "  make setup    - Create virtual environment and install dependencies"
	@echo "  make install  - Install dependencies (requires setup first)"
	@echo "  make run      - Run the main application"
	@echo "  make test     - Run all tests"
	@echo "  make clean    - Clean up generated files and cache"
	@echo "  make clean-all - Clean everything including virtual environment"

# Create virtual environment only
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	@echo "Virtual environment created. Run 'make install' to install dependencies."

# Setup virtual environment and install dependencies
.PHONY: setup
setup: venv install

# Install dependencies (requires setup first)
.PHONY: install
install:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	$(VENV_BIN)/pip install -r requirements.txt

# Run the main application
.PHONY: run
run:
	if [ -d "$(VENV)" ]; then \
		set -a; source .env; set +a; $(VENV_BIN)/python main.py; \
	else \
		set -a; source .env; set +a; eval "$$(conda shell.bash hook)" && conda activate python-carmina && python main.py; \
	fi

# Run all tests
.PHONY: test
test:
	@if [ -d "$(VENV)" ]; then \
		$(VENV_BIN)/python -m pytest; \
	else \
		conda run -n python-carmina --no-capture-output python -m pytest; \
	fi

# Clean up generated files and cache
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

# Clean everything including virtual environment
.PHONY: clean-all
clean-all: clean
	rm -rf $(VENV)