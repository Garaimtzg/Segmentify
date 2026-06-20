# Customer Segmentation (RFM + Clustering)

Engineering-grade customer segmentation over real transactional data (UCI
**Online Retail** dataset). The deliverable is a modular Python package with a
reproducible ETL pipeline, RFM feature engineering, unsupervised modeling
(K-Means), dimensionality reduction (PCA), visualizations, a business analysis,
and a lightweight Streamlit dashboard.

> Status: scaffolding (Milestone 1). Subsequent milestones add the ETL, RFM,
> modeling, visualization, analysis, CLI and dashboard.

## Project layout

```
customer-segmentation/
├── config/config.yaml        # paths, snapshot_date, k_range, seed, ...
├── data/{raw,interim,processed}/
├── src/segmentation/         # the package (etl, features, models, viz, ...)
├── reports/figures/          # generated PNGs
└── tests/                    # pytest suite
```

## Setup (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Or simply: `make install`.

## Dataset

This project uses the **UCI Online Retail** dataset (`.xlsx`). It is **not**
committed to the repo. Download it and place it at:

```
data/raw/Online Retail.xlsx
```

(The exact filename is configurable in `config/config.yaml` →
`dataset.raw_filename`.) If the file is missing, the ETL stops and asks for it —
it never fabricates data.

## Common commands

```bash
make lint        # ruff check
make test        # pytest
make pipeline    # full flow: etl -> features -> train -> viz -> analysis
make dashboard   # launch the Streamlit app
make clean       # remove interim/processed data and figures
```

> On Windows (no `make`), run the underlying commands directly, e.g.
> `ruff check .`, `pytest`, `python -m segmentation.cli all`.

## Reproducibility

Everything stochastic (KMeans, PCA) is seeded via `seed` in `config/config.yaml`.
No paths are hardcoded — all configuration lives in that file.
