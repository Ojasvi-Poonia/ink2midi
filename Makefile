.PHONY: install install-dev download prepare train-detector train-detector-fast train-gnn train-gnn-fast train-all train-all-fast inference evaluate test lint clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

download:
	python scripts/download_data.py

prepare:
	python scripts/prepare_data.py

train-detector:
	python scripts/train_detector.py --device cuda --batch 16

train-detector-fast:
	python scripts/train_detector.py --device cuda --batch 16 --fast

train-gnn:
	python scripts/train_gnn.py --device cuda --batch-size 32

train-gnn-fast:
	python scripts/train_gnn.py --device cuda --batch-size 64 --fast

train-all: train-detector train-gnn

train-all-fast: train-detector-fast train-gnn-fast

inference:
	python scripts/run_inference.py --device cuda

evaluate:
	python scripts/evaluate.py --device cuda

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
