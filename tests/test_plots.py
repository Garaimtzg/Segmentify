"""Tests for visualization helpers (Milestone 6).

Only the pure PCA projection is asserted numerically; the plotting functions are
smoke-tested by the pipeline (`make viz`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from segmentation.models.selection import scale_features
from segmentation.viz.plots import project_pca


def test_project_pca_shape_and_variance() -> None:
    rng = np.random.default_rng(0)
    rfm = pd.DataFrame(
        {
            "Recency": rng.integers(1, 365, size=100),
            "Frequency": rng.integers(1, 50, size=100),
            "Monetary": rng.uniform(10, 10000, size=100),
        }
    )
    X = scale_features(rfm)

    coords, ev = project_pca(X, n_components=2, random_state=42)
    assert coords.shape == (100, 2)
    assert ev.shape == (2,)
    # Explained-variance ratios are in [0, 1] and ordered descending.
    assert np.all((ev >= 0) & (ev <= 1))
    assert ev[0] >= ev[1]
