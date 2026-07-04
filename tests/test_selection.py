"""Tests for scaling and K selection (Milestone 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from segmentation.models.selection import scale_features, sweep_k


def test_scale_features_standardized() -> None:
    rfm = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4, 5],
            "Recency": [1, 10, 50, 100, 300],
            "Frequency": [1, 2, 5, 10, 200],
            "Monetary": [10.0, 100.0, 500.0, 1000.0, 50000.0],
        }
    )
    X = scale_features(rfm)
    assert X.shape == (5, 3)
    # StandardScaler -> zero mean, unit variance per column.
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(X.std(axis=0), 1.0, atol=1e-7)


def test_scale_features_log1p_only() -> None:
    rfm = pd.DataFrame(
        {
            "Recency": [0, 9],
            "Frequency": [1, 99],
            "Monetary": [0.0, 999.0],
        }
    )
    X = scale_features(rfm, standardize=False)
    expected = np.log1p(rfm[["Recency", "Frequency", "Monetary"]].to_numpy(float))
    assert np.allclose(X, expected)


def test_sweep_k_metrics_and_best_k() -> None:
    # Three well-separated blobs -> silhouette should peak at K=3.
    X, _ = make_blobs(
        n_samples=150,
        centers=[[0, 0, 0], [10, 10, 10], [20, 0, 20]],
        cluster_std=0.5,
        random_state=42,
    )
    result = sweep_k(X, k_range=[2, 3, 4, 5], random_state=42)

    assert result.k_values == [2, 3, 4, 5]
    assert len(result.inertias) == 4
    assert len(result.silhouettes) == 4
    # Inertia is non-increasing as K grows.
    assert all(
        earlier >= later
        for earlier, later in zip(
            result.inertias, result.inertias[1:], strict=False
        )
    )
    assert result.best_k == 3
