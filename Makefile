# VESLINT - Makefile
# Provides convenient commands for development and deployment

.PHONY: help install install-dev test lint format clean build docker run smoke-test data prep train evaluate all

# Default target
help:
	@echo "VESLINT - Available Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install       Install the package and dependencies"
	@echo "  install-dev   Install with development dependencies"
	@echo "  clean         Clean up build artifacts and cache files"
	@echo ""
	@echo "Development Commands:"
	@echo "  test          Run all tests"
	@echo "  lint          Run code linting (flake8)"
	@echo "  format        Format code with black"
	@echo "  smoke-test    Run smoke tests to verify installation"
	@echo ""
	@echo "Data Commands:"
	@echo "  data          Download sample AIS data (if available)"
	@echo "  prep          Preprocess data for training"
	@echo ""
	@echo "Model Commands:"
	@echo "  train         Train the model pipeline"
	@echo "  train-test    Train in test mode with sample data"
	@echo "  evaluate      Evaluate trained model"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker        Build Docker image"
	@echo "  docker-run    Run Docker container"
	@echo "  docker-dev    Run Docker with development setup"
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  run           Run complete pipeline with default settings"
	@echo "  all           Install, test, and run complete pipeline"

# Variables
PYTHON := python
PIP := pip
DOCKER_IMAGE := VESLINT
DOCKER_TAG := latest

# Setup Commands
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	$(PIP) install pytest pytest-cov black flake8 pre-commit
	pre-commit install

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete

# Development Commands
test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ main.py --max-line-length=100 --extend-ignore=E203,W503

format:
	black src/ tests/ main.py --line-length=100

smoke-test:
	$(PYTHON) main.py --smoke-tests-only

# Data Commands
data:
	@echo "Please download AIS data from https://hub.marinecadastre.gov/pages/vesseltraffic"
	@echo "Place ZIP files in the data/raw/ directory"
	mkdir -p data/raw data/processed data/models

prep:
	@echo "Data preprocessing will be done during training pipeline"
	@echo "Use 'make train' to run the complete pipeline"

# Model Commands
train:
	@if [ -z "$(ZIP_FILE)" ] || [ -z "$(CSV_FILE)" ]; then \
		echo "Error: Please specify ZIP_FILE and CSV_FILE"; \
		echo "Usage: make train ZIP_FILE=path/to/data.zip CSV_FILE=data.csv"; \
		exit 1; \
	fi
	$(PYTHON) main.py --zip-file $(ZIP_FILE) --csv-name $(CSV_FILE)

train-test:
	@if [ -z "$(ZIP_FILE)" ] || [ -z "$(CSV_FILE)" ]; then \
		echo "Error: Please specify ZIP_FILE and CSV_FILE"; \
		echo "Usage: make train-test ZIP_FILE=path/to/data.zip CSV_FILE=data.csv"; \
		exit 1; \
	fi
	$(PYTHON) main.py --zip-file $(ZIP_FILE) --csv-name $(CSV_FILE) --test-mode

evaluate:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: Please specify MODEL_PATH"; \
		echo "Usage: make evaluate MODEL_PATH=data/models/ensemble_model.joblib"; \
		exit 1; \
	fi
	$(PYTHON) main.py --load-model $(MODEL_PATH)

# Docker Commands
docker:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	docker run --rm -it \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/plots:/app/plots \
		-v $(PWD)/logs:/app/logs \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-dev:
	docker-compose -f docker-compose.yml --profile dev up -d

docker-stop:
	docker-compose -f docker-compose.yml down

# Pipeline Commands
run:
	@if [ -z "$(ZIP_FILE)" ] || [ -z "$(CSV_FILE)" ]; then \
		echo "Error: Please specify ZIP_FILE and CSV_FILE"; \
		echo "Usage: make run ZIP_FILE=path/to/data.zip CSV_FILE=data.csv"; \
		exit 1; \
	fi
	$(PYTHON) main.py --zip-file $(ZIP_FILE) --csv-name $(CSV_FILE) --log-level INFO

run-config:
	@if [ -z "$(CONFIG_FILE)" ]; then \
		echo "Error: Please specify CONFIG_FILE"; \
		echo "Usage: make run-config CONFIG_FILE=config.yaml"; \
		exit 1; \
	fi
	$(PYTHON) main.py --config $(CONFIG_FILE)

# Complete workflow
all: install test smoke-test
	@echo "Setup complete! Ready to run pipeline."
	@echo "Use 'make run ZIP_FILE=your_data.zip CSV_FILE=your_data.csv' to start"

# Utility Commands
check-deps:
	$(PIP) check

update-deps:
	$(PIP) install --upgrade -r requirements.txt

notebook:
	jupyter notebook notebooks/

serve-docs:
	@echo "Serving documentation..."
	$(PYTHON) -m http.server 8080 -d docs/

# Development workflow shortcuts
dev-setup: install-dev
	mkdir -p data/raw data/processed data/models plots logs
	@echo "Development environment ready!"

dev-test: format lint test

# CI/CD simulation
ci: install dev-test smoke-test
	@echo "CI pipeline completed successfully!"

# Production deployment preparation
prod-check: clean install test lint
	@echo "Production readiness check completed!"

# Performance profiling
profile:
	@if [ -z "$(ZIP_FILE)" ] || [ -z "$(CSV_FILE)" ]; then \
		echo "Error: Please specify ZIP_FILE and CSV_FILE for profiling"; \
		exit 1; \
	fi
	$(PYTHON) -m cProfile -o profile.prof main.py --zip-file $(ZIP_FILE) --csv-name $(CSV_FILE) --test-mode
	@echo "Profile saved to profile.prof. Use 'python -m pstats profile.prof' to analyze"

# Memory usage analysis
memory-profile:
	@echo "Installing memory profiler..."
	$(PIP) install memory-profiler
	@if [ -z "$(ZIP_FILE)" ] || [ -z "$(CSV_FILE)" ]; then \
		echo "Error: Please specify ZIP_FILE and CSV_FILE for memory profiling"; \
		exit 1; \
	fi
	$(PYTHON) -m memory_profiler main.py --zip-file $(ZIP_FILE) --csv-name $(CSV_FILE) --test-mode

# Security check
security-check:
	$(PIP) install safety bandit
	safety check
	bandit -r src/ -f json -o security-report.json || true
	@echo "Security check completed. See security-report.json for details."

# Example data simulation (for testing when no real data available)
simulate-data:
	$(PYTHON) -c "
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile

# Create synthetic AIS data
np.random.seed(42)
n_records = 10000

data = {
    'MMSI': np.random.choice([123456789, 987654321, 111222333, 444555666], n_records),
    'LAT': np.random.uniform(25, 48, n_records),
    'LON': np.random.uniform(-125, -70, n_records),
    'SOG': np.random.uniform(0, 30, n_records),
    'COG': np.random.uniform(0, 360, n_records),
    'VesselType': np.random.choice([30, 31, 37, 52, 70], n_records),
    'BaseDateTime': pd.date_range('2024-01-01', periods=n_records, freq='5min'),
    'Length': np.random.uniform(20, 400, n_records),
    'Width': np.random.uniform(5, 60, n_records),
    'Draft': np.random.uniform(1, 20, n_records)
}

df = pd.DataFrame(data)
Path('data/raw').mkdir(parents=True, exist_ok=True)
df.to_csv('data/raw/synthetic_ais_data.csv', index=False)

# Create ZIP file
with zipfile.ZipFile('data/raw/synthetic_ais_data.zip', 'w') as zf:
    zf.write('data/raw/synthetic_ais_data.csv', 'synthetic_ais_data.csv')

print('Synthetic data created: data/raw/synthetic_ais_data.zip')
print('Use: make run ZIP_FILE=data/raw/synthetic_ais_data.zip CSV_FILE=synthetic_ais_data.csv')
"

# Help for specific commands
help-train:
	@echo "Training Commands Help:"
	@echo ""
	@echo "make train ZIP_FILE=path/to/data.zip CSV_FILE=data.csv"
	@echo "  - Train model with full dataset"
	@echo ""
	@echo "make train-test ZIP_FILE=path/to/data.zip CSV_FILE=data.csv"
	@echo "  - Train model with sample data (faster for testing)"
	@echo ""
	@echo "make evaluate MODEL_PATH=data/models/ensemble_model.joblib"
	@echo "  - Evaluate existing trained model"
	@echo ""
	@echo "Example workflow:"
	@echo "  1. make simulate-data  # Create test data"
	@echo "  2. make train-test ZIP_FILE=data/raw/synthetic_ais_data.zip CSV_FILE=synthetic_ais_data.csv"
	@echo "  3. make evaluate MODEL_PATH=data/models/ensemble_model.joblib"