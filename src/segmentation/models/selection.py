"""Model selection: scale the RFM matrix and choose K.

Pure, testable functions (no I/O, no plotting). The preprocessing follows the
brief's recommendation: ``log1p`` to correct the strong right-skew of Frequency
and Monetary, then :class:`~sklearn.preprocessing.StandardScaler`. K is swept
over a configurable range, computing the **Elbow** (inertia) and the
**silhouette** score for each K.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RFM_FEATURES = ["Recency", "Frequency", "Monetary"]


def scale_features(
    rfm: pd.DataFrame,
    *,
    features: Sequence[str] = RFM_FEATURES,
    log1p: bool = True,
    standardize: bool = True,
) -> np.ndarray:
    """Return the scaled feature matrix used for clustering and PCA.

    Args:
        rfm: RFM table containing at least the ``features`` columns.
        features: Columns to use (defaults to Recency/Frequency/Monetary).
        log1p: Apply ``log1p`` to reduce skew before standardizing.
        standardize: Apply ``StandardScaler`` (zero mean, unit variance).

    Returns:
        A ``(n_customers, n_features)`` float array. The transform is
        deterministic, so callers can recompute it consistently.
    """
    X = rfm.loc[:, list(features)].to_numpy(dtype=float)
    if log1p:
        X = np.log1p(X)
    if standardize:
        X = StandardScaler().fit_transform(X)
    return X


@dataclass(frozen=True)
class KSelectionResult:
    """Metrics from sweeping K, plus the silhouette-optimal K."""

    k_values: list[int]
    inertias: list[float]
    silhouettes: list[float]

    @property
    def best_k(self) -> int:
        """K with the highest silhouette score."""
        return self.k_values[int(np.argmax(self.silhouettes))]

    @property
    def best_silhouette(self) -> float:
        return max(self.silhouettes)


def sweep_k(
    X: np.ndarray,
    *,
    k_range: Sequence[int],
    random_state: int = 42,
) -> KSelectionResult:
    """Fit KMeans for each K and collect inertia + silhouette.

    Args:
        X: Scaled feature matrix.
        k_range: Iterable of K values to try (each must be >= 2 and < n_samples).
        random_state: Seed for KMeans reproducibility.

    Returns:
        A :class:`KSelectionResult` with one metric per K.

    Raises:
        ValueError: if ``k_range`` is empty.
    """
    ks = [int(k) for k in k_range]
    if not ks:
        raise ValueError("k_range must contain at least one value.")

    inertias: list[float] = []
    silhouettes: list[float] = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X, labels)))

    return KSelectionResult(k_values=ks, inertias=inertias, silhouettes=silhouettes)
