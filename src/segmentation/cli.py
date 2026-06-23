"""Command-line entry point for the segmentation pipeline.

Subcommands are added milestone by milestone. Currently available:

* ``etl`` — extract the raw dataset, clean it and persist the result.

Run as ``python -m segmentation.cli <command>`` or via the ``segmentation``
console script.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from .analysis.personas import build_segment_profiles
from .config import Config, load_config
from .etl.extract import read_raw_transactions
from .etl.load import load_parquet, save_csv, save_parquet
from .etl.transform import clean_transactions
from .features.rfm import compute_rfm
from .models.clustering import add_cluster_labels, fit_kmeans
from .models.selection import scale_features, sweep_k
from .viz.plots import (
    plot_elbow,
    plot_pca_scatter,
    plot_pca_scatter_3d,
    plot_rfm_profiles,
    plot_silhouette,
    project_pca,
)

app = typer.Typer(
    add_completion=False,
    help="Customer segmentation pipeline (RFM + clustering).",
)
logger = logging.getLogger("segmentation")

ConfigOption = typer.Option(
    None,
    "--config",
    "-c",
    help="Path to config.yaml (defaults to the project's config/config.yaml).",
)


@app.callback()
def main() -> None:
    """Customer segmentation pipeline (RFM + clustering)."""


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load(config: Optional[Path]) -> Config:
    return load_config(config) if config is not None else load_config()


def run_etl(cfg: Config) -> Path:
    """Execute the ETL stage and return the path of the cleaned artifact."""
    logger.info("Reading raw dataset from %s", cfg.raw_dataset_path)
    raw = read_raw_transactions(
        cfg.raw_dataset_path, sheet_name=cfg.dataset.get("sheet_name", 0)
    )
    logger.info("Raw rows: %d", len(raw))

    clean = clean_transactions(
        raw,
        cancellation_prefix=cfg.etl["cancellation_prefix"],
        min_quantity=cfg.etl["min_quantity"],
        min_unit_price=cfg.etl["min_unit_price"],
    )
    logger.info(
        "Clean rows: %d (customers: %d)",
        len(clean),
        clean["CustomerID"].nunique(),
    )

    out = save_parquet(clean, cfg.transactions_clean_path)
    logger.info("Wrote cleaned transactions to %s", out)
    return out


def run_features(cfg: Config) -> Path:
    """Build the RFM matrix from cleaned transactions and persist it."""
    logger.info("Reading cleaned transactions from %s", cfg.transactions_clean_path)
    transactions = load_parquet(cfg.transactions_clean_path)

    snapshot = cfg.rfm.get("snapshot_date")
    rfm = compute_rfm(transactions, snapshot_date=snapshot)
    logger.info(
        "Computed RFM for %d customers (snapshot: %s)",
        len(rfm),
        snapshot if snapshot is not None else "max(InvoiceDate)+1d",
    )

    out = save_parquet(rfm, cfg.rfm_path)
    logger.info("Wrote RFM matrix to %s", out)
    return out


def run_train(cfg: Config) -> Path:
    """Select K, export selection figures, fit KMeans and persist labels.

    Returns the path of the clustered RFM artifact.
    """
    logger.info("Reading RFM matrix from %s", cfg.rfm_path)
    rfm = load_parquet(cfg.rfm_path)

    X = scale_features(
        rfm,
        log1p=cfg.model.get("log1p", True),
        standardize=cfg.model.get("standardize", True),
    )
    result = sweep_k(
        X,
        k_range=cfg.model["k_range"],
        random_state=cfg.model["random_state"],
    )
    for k, inertia, sil in zip(
        result.k_values, result.inertias, result.silhouettes, strict=True
    ):
        logger.info("K=%2d  inertia=%12.1f  silhouette=%.4f", k, inertia, sil)

    configured_k = cfg.model.get("chosen_k")
    chosen_k = int(configured_k) if configured_k else result.best_k
    source = "config.yaml" if configured_k else "best silhouette"
    logger.info(
        "Chosen K = %d (%s; silhouette peaks at K=%d with %.4f)",
        chosen_k,
        source,
        result.best_k,
        result.best_silhouette,
    )

    figures_dir = cfg.paths.figures_dir
    elbow_path = plot_elbow(result, figures_dir / "elbow.png", chosen_k=chosen_k)
    sil_path = plot_silhouette(
        result, figures_dir / "silhouette.png", chosen_k=chosen_k
    )
    logger.info("Wrote figures: %s , %s", elbow_path, sil_path)

    # Fit the final model with the chosen K and persist per-customer labels.
    model = fit_kmeans(
        X, n_clusters=chosen_k, random_state=cfg.model["random_state"]
    )
    clustered = add_cluster_labels(rfm, model.labels_)
    sizes = clustered["Cluster"].value_counts().sort_index().to_dict()
    logger.info("Cluster sizes: %s", sizes)

    out = save_parquet(clustered, cfg.rfm_clustered_path)
    logger.info("Wrote clustered RFM (K=%d) to %s", chosen_k, out)
    return out


def run_viz(cfg: Config) -> list[Path]:
    """Generate PCA scatters (2D/3D) and the RFM profile heatmap."""
    logger.info("Reading clustered RFM from %s", cfg.rfm_clustered_path)
    clustered = load_parquet(cfg.rfm_clustered_path)

    X = scale_features(
        clustered,
        log1p=cfg.model.get("log1p", True),
        standardize=cfg.model.get("standardize", True),
    )
    labels = clustered["Cluster"].to_numpy()
    seed = cfg.model["random_state"]
    figures_dir = cfg.paths.figures_dir

    coords2, ev2 = project_pca(X, n_components=2, random_state=seed)
    logger.info("PCA 2D explained variance: %s", (ev2 * 100).round(1).tolist())
    pca2_path = plot_pca_scatter(
        coords2, labels, figures_dir / "pca_2d.png", explained_variance=ev2
    )

    coords3, ev3 = project_pca(X, n_components=3, random_state=seed)
    pca3_path = plot_pca_scatter_3d(
        coords3, labels, figures_dir / "pca_3d.png", explained_variance=ev3
    )

    profiles_path = plot_rfm_profiles(clustered, figures_dir / "rfm_profiles.png")

    paths = [pca2_path, pca3_path, profiles_path]
    logger.info("Wrote figures: %s", ", ".join(str(p) for p in paths))
    return paths


def run_analysis(cfg: Config) -> Path:
    """Map clusters to personas and export the segment profiles table."""
    logger.info("Reading clustered RFM from %s", cfg.rfm_clustered_path)
    clustered = load_parquet(cfg.rfm_clustered_path)

    persona_names = cfg.raw.get("personas") or {}
    profiles = build_segment_profiles(clustered, persona_names=persona_names)

    for _, row in profiles.iterrows():
        logger.info(
            "Cluster %s -> %-20s size=%4d (%.1f%%)  R=%.1f F=%.1f M=%.1f",
            row["Cluster"], row["Persona"], row["Size"], row["Share"],
            row["Recency"], row["Frequency"], row["Monetary"],
        )

    out = save_csv(profiles, cfg.segment_profiles_path)
    logger.info("Wrote segment profiles to %s", out)
    return out


@app.command()
def etl(config: Optional[Path] = ConfigOption) -> None:
    """Extract, clean and persist the raw transactions."""
    _setup_logging()
    cfg = _load(config)
    run_etl(cfg)


@app.command()
def features(config: Optional[Path] = ConfigOption) -> None:
    """Build the RFM feature matrix from cleaned transactions."""
    _setup_logging()
    cfg = _load(config)
    run_features(cfg)


@app.command()
def train(config: Optional[Path] = ConfigOption) -> None:
    """Select K, fit KMeans and persist per-customer cluster labels."""
    _setup_logging()
    cfg = _load(config)
    run_train(cfg)


@app.command()
def viz(config: Optional[Path] = ConfigOption) -> None:
    """Generate PCA scatters and the RFM profile figure."""
    _setup_logging()
    cfg = _load(config)
    run_viz(cfg)


@app.command()
def analysis(config: Optional[Path] = ConfigOption) -> None:
    """Map clusters to personas and export the segment profiles table."""
    _setup_logging()
    cfg = _load(config)
    run_analysis(cfg)


@app.command("all")
def run_all(config: Optional[Path] = ConfigOption) -> None:
    """Run the full pipeline: etl -> features -> train -> viz -> analysis."""
    _setup_logging()
    cfg = _load(config)
    logger.info("===== Pipeline start =====")
    run_etl(cfg)
    run_features(cfg)
    run_train(cfg)
    run_viz(cfg)
    run_analysis(cfg)
    logger.info("===== Pipeline complete =====")


if __name__ == "__main__":
    app()
