.PHONY: help install test lint format clean build docker run-docker docs

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean generated files"
	@echo "  make build      - Build package"
	@echo "  make docker     - Build Docker image"
	@echo "  make run-docker - Run Docker container"
	@echo "  make docs       - Generate documentation"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .[dev]
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -x --tb=short

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/
	autopep8 --in-place --recursive src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf models/*.pth results/*.png data/*.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python -m build

docker:
	docker build -t calabi-yau-ml:latest .

run-docker:
	docker run -it --rm \
		-v $(PWD)/data:/data \
		-v $(PWD)/models:/models \
		-v $(PWD)/results:/results \
		-p 6006:6006 \
		calabi-yau-ml:latest

docs:
	cd docs && make html

benchmark:
	python benchmarks/benchmark_models.py
	python benchmarks/benchmark_data.py

profile:
	python -m cProfile -o profile.stats run_experiment.py --task regression --epochs 10
	python -m pstats profile.stats

security:
	bandit -r src/ -ll
	safety check

update-deps:
	pip-compile requirements.in -o requirements.txt --upgrade

serve:
	python -m src.api.serve --host 0.0.0.0 --port 8000

train-quick:
	python run_experiment.py --task regression --epochs 20 --n_samples 1000

train-full:
	python run_experiment.py --task regression --epochs 100 --n_samples 5000

train-cicy:
	python src/train_real_data.py --target h11_prediction --epochs 50

deploy-test:
	twine upload --repository testpypi dist/*

deploy-prod:
	twine upload dist/*
