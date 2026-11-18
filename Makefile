# BabyJoke joke-type video classification
# ========================================
#
# Usage:
#   make help
#   make setup
#   make multimodal | 3d-cnn | tan | cnn-lstm
#   make ablation
#   make analysis
#   make test

.PHONY: help all cnn-lstm 3d-cnn tan multimodal ablation analysis clean clean-all setup test test-setup check-results

help:
	@echo "BabyJoke five-class joke-type classification"
	@echo "============================================"
	@echo "  setup       - pip install -r requirements.txt"
	@echo "  cnn-lstm    - Run CNN-LSTM nested ChildID CV"
	@echo "  3d-cnn      - Run 3D-CNN nested ChildID CV"
	@echo "  tan         - Run TAN nested ChildID CV"
	@echo "  multimodal  - Run multi-cue nested ChildID CV"
	@echo "  ablation    - Run seven multi-cue ablation variants"
	@echo "  analysis    - Execute analysis notebook (reference_results or outputs/)"
	@echo "  test        - Schema / protocol unit tests (no GPU)"
	@echo "  clean       - Clear caches"
	@echo "  clean-all   - Clear caches and outputs/"

setup:
	@echo "Installing dependencies from requirements.txt ..."
	pip install -r requirements.txt
	@echo "Setup complete."

all: cnn-lstm 3d-cnn tan multimodal analysis
	@echo "All models finished. Results under outputs/ (or use analysis/reference_results/)."

cnn-lstm:
	python scripts/run_cnn_lstm.py

3d-cnn:
	python scripts/run_3d_cnn.py

tan:
	python scripts/run_tan.py

multimodal:
	python scripts/run_multimodal.py

ablation:
	python scripts/run_multimodal_ablation.py

analysis:
	@if [ -f "analysis/reference_results/all_models_performance.csv" ]; then \
		echo "Using analysis/reference_results/ (notebook default)"; \
	elif [ -f "outputs/all_models_performance.csv" ]; then \
		echo "Using outputs/all_models_performance.csv"; \
	else \
		echo "Error: need analysis/reference_results/ or outputs/all_models_performance.csv"; \
		exit 1; \
	fi
	jupyter nbconvert --execute --to notebook --inplace analysis/cross_validation_analysis.ipynb
	@echo "Analysis notebook executed."

test:
	python tests/test_artifact_schema.py
	python tests/test_nested_refit_schema.py

clean:
	rm -rf data/cache/* __pycache__/ */__pycache__/ .pytest_cache/ *.pyc *.pyo 2>/dev/null || true
	@echo "Cache cleaned."

clean-all: clean
	rm -rf outputs/* analysis/figures/* analysis/outputs/* 2>/dev/null || true
	@echo "Outputs cleaned (reference_results retained)."

test-setup:
	@python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
	@python -c "import cv2; print('OpenCV', cv2.__version__)"
	@python -c "import sklearn; print('sklearn', sklearn.__version__)"

check-results:
	@if [ -f "analysis/reference_results/all_models_performance.csv" ]; then \
		echo "reference_results CSV present"; \
		wc -l analysis/reference_results/all_models_performance.csv; \
	fi
	@if [ -f "outputs/all_models_performance.csv" ]; then \
		echo "outputs CSV present"; \
		wc -l outputs/all_models_performance.csv; \
	fi
