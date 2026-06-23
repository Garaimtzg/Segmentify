# Customer Segmentation — orchestration targets
# Target platform: Linux (see README for venv setup).

# Interpreter used to create the venv. Override if needed, e.g.
#   make install PYTHON=python3.11
# Ubuntu 26.04 ships Python 3.14 as `python3`, which lacks wheels for the pinned
# stack, so we default to 3.12.
PYTHON ?= python3.12
VENV   := .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python
PIP    := $(BIN)/pip

.PHONY: help install lint format test etl features train viz analysis pipeline dashboard clean

help:
	@echo "Targets:"
	@echo "  install    create venv and install dependencies (editable)"
	@echo "  lint       ruff check"
	@echo "  format     ruff format"
	@echo "  test       pytest"
	@echo "  etl        run the ETL stage only"
	@echo "  features   build the RFM feature matrix"
	@echo "  train      select K and fit KMeans"
	@echo "  viz        generate PCA + profile figures"
	@echo "  analysis   map clusters to personas"
	@echo "  pipeline   run the full flow (etl -> features -> train -> viz -> analysis)"
	@echo "  dashboard  launch the Streamlit app"
	@echo "  clean      remove interim/processed data and figures"

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

lint:
	$(BIN)/ruff check .

format:
	$(BIN)/ruff format .

test:
	$(BIN)/pytest

etl:
	$(PY) -m segmentation.cli etl

features:
	$(PY) -m segmentation.cli features

train:
	$(PY) -m segmentation.cli train

viz:
	$(PY) -m segmentation.cli viz

analysis:
	$(PY) -m segmentation.cli analysis

pipeline:
	$(PY) -m segmentation.cli all

dashboard:
	$(BIN)/streamlit run src/segmentation/dashboard/app.py

clean:
	rm -rf data/interim/* data/processed/* reports/figures/* reports/segment_profiles.csv
