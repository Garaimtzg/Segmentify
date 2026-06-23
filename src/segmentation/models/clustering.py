"""Clustering: fit KMeans on the scaled RFM matrix and label each customer.

Pure, testable functions (no I/O). ``random_state`` is always passed so results
are reproducible. Scaling lives in :mod:`segmentation.models.selection`
(``scale_features``); this module consumes the already-scaled matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def fit_kmeans(
    X: np.ndarray,
    *,
    n_clusters: int,
    random_state: int = 42,
) -> KMeans:
    """Fit a KMeans model on the scaled feature matrix.

    Args:
        X: Scaled feature matrix (e.g. from ``scale_features``).
        n_clusters: Number of clusters (K).
        random_state: Seed for reproducibility.

    Returns:
        The fitted :class:`~sklearn.cluster.KMeans` model.
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X)
    return model


def add_cluster_labels(
    rfm: pd.DataFrame,
    labels: np.ndarray,
    *,
    column: str = "Cluster",
) -> pd.DataFrame:
    """Return a copy of ``rfm`` with a cluster-label column attached.

    The labels must be aligned row-by-row with ``rfm`` (same order/length).

    Raises:
        ValueError: if the number of labels does not match the rows of ``rfm``.
    """
    if len(labels) != len(rfm):
        raise ValueError(
            f"labels length ({len(labels)}) does not match rfm rows ({len(rfm)})."
        )
    out = rfm.copy()
    out[column] = np.asarray(labels).astype(int)
    return out
