# Makefile for Multimodal Deep Learning for Joke Classification
# =====================================================================
# 
# This Makefile provides convenient targets to run each model and generate
# comprehensive analysis. All outputs are saved to outputs/ directory.
#
# Usage:
#   make help          - Show available targets
#   make all           - Run all models and analysis
#   make cnn-lstm      - Run CNN-LSTM model only
#   make 3d-cnn        - Run 3D-CNN model only
#   make tan           - Run Temporal Attention Network only
#   make multimodal    - Run Multimodal model only
#   make analysis      - Generate cross-validation analysis
#   make clean         - Clean cache and temporary files
#   make clean-all     - Clean all outputs and cache

.PHONY: help all cnn-lstm 3d-cnn tan multimodal analysis clean clean-all install setup

# Default target
help:
	@echo "Multimodal Deep Learning Framework for Joke Classification"
	@echo "=========================================================="
	@echo ""
	@echo "Available targets:"
	@echo "  help          - Show this help message"
	@echo "  setup         - Install dependencies and setup environment"
	@echo "  all           - Run all models and generate analysis"
	@echo "  cnn-lstm      - Run CNN-LSTM model (10-fold CV)"
	@echo "  3d-cnn        - Run 3D-CNN model (10-fold CV)"
	@echo "  tan           - Run Temporal Attention Network (10-fold CV)"
	@echo "  multimodal    - Run Multimodal model (10-fold CV)"
	@echo "  analysis      - Generate cross-validation analysis"
	@echo "  clean         - Clean cache and temporary files"
	@echo "  clean-all     - Clean all outputs and cache"
	@echo ""
	@echo "Examples:"
	@echo "  make setup && make all"
	@echo "  make cnn-lstm multimodal"
	@echo "  make clean && make analysis"

# Setup environment and install dependencies
setup:
	@echo "Setting up environment..."
	@if [ ! -f "requirment.txt" ]; then \
		echo "Error: requirment.txt not found"; \
		exit 1; \
	fi
	@echo "Installing Python dependencies..."
	pip install -r requirment.txt
	@echo "Installing MediaPipe for multimodal features..."
	pip install mediapipe || pip install mediapipe==0.10.14
	@echo "Setup complete!"

# Run all models and analysis
all: cnn-lstm 3d-cnn tan multimodal analysis
	@echo "All models completed and analysis generated!"
	@echo "Results available in outputs/ directory"

# Individual model targets
cnn-lstm:
	@echo "Running CNN-LSTM model..."
	python run_cnn_lstm.py
	@echo "CNN-LSTM completed!"

3d-cnn:
	@echo "🚀 Running 3D-CNN model..."
	python run_3d_cnn.py
	@echo "3D-CNN completed!"

tan:
	@echo "Running Temporal Attention Network..."
	python run_tan.py
	@echo "TAN completed!"

multimodal:
	@echo "🚀 Running Multimodal model..."
	python run_multimodal.py
	@echo "Multimodal completed!"

# Generate cross-validation analysis
analysis:
	@echo "Generating cross-validation analysis..."
	@if [ ! -f "outputs/all_models_performance.csv" ]; then \
		echo "Error: all_models_performance.csv not found. Run models first."; \
		exit 1; \
	fi
	@if [ ! -f "analysis/cross_validation_analysis.ipynb" ]; then \
		echo "Error: cross_validation_analysis.ipynb not found"; \
		exit 1; \
	fi
	jupyter nbconvert --execute --to notebook --inplace analysis/cross_validation_analysis.ipynb
	@echo "Analysis completed! Check analysis/figures/ and analysis/outputs/"

# Clean cache and temporary files
clean:
	@echo "🧹 Cleaning cache and temporary files..."
	rm -rf data/cache/*
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf .pytest_cache/
	rm -rf *.pyc
	rm -rf *.pyo
	@echo "Cache cleaned!"

# Clean all outputs and cache
clean-all: clean
	@echo "🧹 Cleaning all outputs..."
	rm -rf outputs/*
	rm -rf analysis/figures/*
	rm -rf analysis/outputs/*
	@echo "All outputs cleaned!"

# Quick test to verify setup
test-setup:
	@echo "Testing setup..."
	@python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
	@python -c "import cv2; print('OpenCV:', cv2.__version__)"
	@python -c "import numpy as np; print('NumPy:', np.__version__)"
	@python -c "import pandas as pd; print('Pandas:', pd.__version__)"
	@python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
	@python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
	@python -c "import seaborn; print('Seaborn:', seaborn.__version__)"
	@python -c "try: import mediapipe; print('MediaPipe:', mediapipe.__version__); except: print('MediaPipe not installed')"
	@echo "Setup test completed!"

# Check if all models have been run
check-results:
	@echo "Checking results..."
	@if [ -f "outputs/all_models_performance.csv" ]; then \
		echo "Global results CSV found"; \
		lines=$$(wc -l < outputs/all_models_performance.csv); \
		echo "CSV has $$lines lines (expected: 41 with header)"; \
	else \
		echo "Global results CSV not found"; \
	fi
	@for model in CNN_LSTM CNN3D TAN Multimodal; do \
		if [ -d "outputs/$$model" ]; then \
			echo "$$model results directory found"; \
		else \
			echo "$$model results directory not found"; \
		fi; \
	done

# Run specific model combinations
models-rgb: cnn-lstm 3d-cnn tan
	@echo "RGB-based models completed!"

models-behavioral: multimodal
	@echo "Behavioral model completed!"

# Development targets
dev-install:
	@echo "Installing development dependencies..."
	pip install jupyter notebook ipykernel pytest black flake8

format:
	@echo "Formatting code..."
	black *.py models/*.py src/*.py
	@echo "Code formatted!"

lint:
	@echo "🔍 Linting code..."
	flake8 *.py models/*.py src/*.py --max-line-length=100
	@echo "Linting completed!"

# Documentation targets
docs:
	@echo "Generating documentation..."
	@if [ -f "README.md" ]; then \
		echo "README.md found"; \
	else \
		echo "README.md not found"; \
	fi
	@echo "Documentation status: Complete"

# Performance monitoring
monitor:
	@echo "Monitoring system resources..."
	@echo "Memory usage:"
	@ps aux | grep python | head -5
	@echo "Disk usage:"
	@du -sh outputs/ data/cache/ 2>/dev/null || echo "No cache/output found"

# Backup results
backup:
	@echo "Creating backup of results..."
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	backup_dir="backup_$$timestamp"; \
	mkdir -p "$$backup_dir"; \
	cp -r outputs/ "$$backup_dir/" 2>/dev/null || echo "No outputs to backup"; \
	cp -r analysis/ "$$backup_dir/" 2>/dev/null || echo "No analysis to backup"; \
	echo "Backup created: $$backup_dir"

# Show current status
status:
	@echo "Current Status:"
	@echo "=================="
	@echo "Project structure:"
	@ls -la | grep -E "(README|Makefile|requirment|run_|config|models|src|analysis|outputs)"
	@echo ""
	@echo "Results status:"
	@make check-results
	@echo ""
	@echo "Cache status:"
	@if [ -d "data/cache" ]; then \
		cache_size=$$(du -sh data/cache 2>/dev/null | cut -f1); \
		cache_files=$$(find data/cache -name "*.pkl" 2>/dev/null | wc -l); \
		echo "Cache: $$cache_size ($$cache_files files)"; \
	else \
		echo "No cache directory"; \
	fi
