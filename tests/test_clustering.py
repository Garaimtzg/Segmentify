"""Tests for clustering (Milestone 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from segmentation.models.clustering import add_cluster_labels, fit_kmeans
from segmentation.models.selection import scale_features


def _blob_rfm(k: int = 3, n: int = 150) -> pd.DataFrame:
    """Build an RFM-like frame with `k` well-separated groups."""
    X, _ = make_blobs(
        n_samples=n,
        centers=[[1, 1, 1], [50, 20, 200], [120, 80, 5000]][:k],
        cluster_std=0.5,
        random_state=42,
    )
    X = np.abs(X)  # RFM features are non-negative
    return pd.DataFrame(
        {
            "CustomerID": range(1, len(X) + 1),
            "Recency": X[:, 0],
            "Frequency": X[:, 1],
            "Monetary": X[:, 2],
        }
    )


def test_every_customer_gets_a_label() -> None:
    rfm = _blob_rfm(k=3)
    scaled = scale_features(rfm)
    model = fit_kmeans(scaled, n_clusters=3, random_state=42)
    clustered = add_cluster_labels(rfm, model.labels_)

    assert "Cluster" in clustered.columns
    assert clustered["Cluster"].notna().all()
    assert len(clustered) == len(rfm)


def test_number_of_clusters_matches_k() -> None:
    rfm = _blob_rfm(k=3)
    scaled = scale_features(rfm)
    model = fit_kmeans(scaled, n_clusters=3, random_state=42)
    clustered = add_cluster_labels(rfm, model.labels_)

    assert clustered["Cluster"].nunique() == 3
    assert set(clustered["Cluster"].unique()) == {0, 1, 2}


def test_labels_are_reproducible() -> None:
    rfm = _blob_rfm(k=3)
    scaled = scale_features(rfm)
    a = fit_kmeans(scaled, n_clusters=4, random_state=42).labels_
    b = fit_kmeans(scaled, n_clusters=4, random_state=42).labels_
    assert np.array_equal(a, b)


def test_label_count_mismatch_raises() -> None:
    rfm = _blob_rfm(k=3)
    with pytest.raises(ValueError, match="does not match"):
        add_cluster_labels(rfm, np.array([0, 1, 2]))
