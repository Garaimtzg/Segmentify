"""Plotting helpers for the pipeline.

Uses a non-interactive (``Agg``) backend so figures render headless (e.g. under
WSL / CI). Each function saves a PNG and returns its path.

Covers the selection figures (Elbow, silhouette) and the post-clustering
visualizations (PCA scatter in 2D/3D and the RFM profile heatmap).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

matplotlib.use("Agg")  # headless backend; must be set before pyplot import

import matplotlib.pyplot as plt  # noqa: E402

from ..models.selection import KSelectionResult  # noqa: E402

RFM_FEATURES = ["Recency", "Frequency", "Monetary"]


def _ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# K-selection figures
# --------------------------------------------------------------------------- #
def plot_elbow(
    result: KSelectionResult, path: str | Path, chosen_k: int | None = None
) -> Path:
    """Plot inertia vs K (the Elbow curve) and mark the chosen K."""
    marked_k = result.best_k if chosen_k is None else chosen_k
    out = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.k_values, result.inertias, marker="o", color="#1f77b4")
    ax.axvline(marked_k, color="#d62728", linestyle="--", alpha=0.7,
               label=f"chosen K = {marked_k}")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("Elbow method")
    ax.set_xticks(result.k_values)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_silhouette(
    result: KSelectionResult, path: str | Path, chosen_k: int | None = None
) -> Path:
    """Plot silhouette score vs K and mark the chosen K."""
    marked_k = result.best_k if chosen_k is None else chosen_k
    out = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.k_values, result.silhouettes, marker="o", color="#2ca02c")
    ax.axvline(marked_k, color="#d62728", linestyle="--", alpha=0.7,
               label=f"chosen K = {marked_k}")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette analysis")
    ax.set_xticks(result.k_values)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# PCA + cluster profiles
# --------------------------------------------------------------------------- #
def project_pca(
    X: np.ndarray, n_components: int = 2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Project the scaled feature matrix onto its principal components.

    Returns:
        ``(coords, explained_variance_ratio)`` where ``coords`` has shape
        ``(n_samples, n_components)``.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def _axis_label(component: int, explained_variance: np.ndarray | None) -> str:
    label = f"PC{component + 1}"
    if explained_variance is not None and component < len(explained_variance):
        label += f" ({explained_variance[component] * 100:.1f}% var)"
    return label


def plot_pca_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    explained_variance: np.ndarray | None = None,
) -> Path:
    """2D scatter of PCA coordinates, colored by cluster label."""
    out = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for i, label in enumerate(sorted(np.unique(labels))):
        mask = labels == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=12, alpha=0.6, color=cmap(i % 10), label=f"Cluster {label}",
        )
    ax.set_xlabel(_axis_label(0, explained_variance))
    ax.set_ylabel(_axis_label(1, explained_variance))
    ax.set_title("Customer segments in PCA space (2D)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_pca_scatter_3d(
    coords: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    explained_variance: np.ndarray | None = None,
) -> Path:
    """3D scatter of PCA coordinates, colored by cluster label."""
    out = _ensure_parent(path)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")
    for i, label in enumerate(sorted(np.unique(labels))):
        mask = labels == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            s=10, alpha=0.6, color=cmap(i % 10), label=f"Cluster {label}",
        )
    ax.set_xlabel(_axis_label(0, explained_variance))
    ax.set_ylabel(_axis_label(1, explained_variance))
    ax.set_zlabel(_axis_label(2, explained_variance))
    ax.set_title("Customer segments in PCA space (3D)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_rfm_profiles(
    clustered: pd.DataFrame,
    path: str | Path,
    *,
    features: Sequence[str] = RFM_FEATURES,
    cluster_col: str = "Cluster",
) -> Path:
    """Heatmap of mean RFM per cluster.

    Cell color is min-max normalized per feature (so columns are comparable
    despite different scales); the annotation shows the raw mean value.
    """
    out = _ensure_parent(path)
    means = clustered.groupby(cluster_col)[list(features)].mean()
    span = (means.max() - means.min()).replace(0, 1)
    norm = (means - means.min()) / span

    fig, ax = plt.subplots(figsize=(7, 0.8 * len(means) + 2))
    im = ax.imshow(norm.to_numpy(), cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(features)), labels=list(features))
    ax.set_yticks(range(len(means)), labels=[f"Cluster {c}" for c in means.index])
    for r in range(len(means)):
        for c in range(len(features)):
            ax.text(
                c, r, f"{means.iat[r, c]:,.1f}",
                ha="center", va="center",
                color="black" if norm.iat[r, c] < 0.6 else "white",
            )
    ax.set_title("RFM profile by cluster (color = per-feature min-max)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalized mean")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
