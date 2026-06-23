"""Tests for persona assignment (Milestone 7)."""

from __future__ import annotations

import pandas as pd
import pytest

from segmentation.analysis.personas import PERSONA_DEFAULTS, build_segment_profiles


def _clustered() -> pd.DataFrame:
    """Four clusters, one per RFM quadrant (recent/valuable)."""
    rows = []
    # Cluster 0: recent + high value -> Champions
    rows += [(c, 10, 12, 9000, 0) for c in range(10)]
    # Cluster 1: recent + low value -> Promising
    rows += [(c, 15, 2, 300, 1) for c in range(10)]
    # Cluster 2: not recent + high value -> At risk
    rows += [(c, 120, 5, 2000, 2) for c in range(10)]
    # Cluster 3: not recent + low value -> Lost
    rows += [(c, 250, 1, 200, 3) for c in range(10)]
    return pd.DataFrame(
        rows, columns=["CustomerID", "Recency", "Frequency", "Monetary", "Cluster"]
    )


def test_profiles_one_row_per_cluster_with_personas() -> None:
    profiles = build_segment_profiles(_clustered())
    assert len(profiles) == 4
    assert set(profiles["Cluster"]) == {0, 1, 2, 3}
    assert set(profiles.columns) == {
        "Cluster", "Persona", "Size", "Share", "Recency", "Frequency", "Monetary"
    }


def test_persona_assignment_matches_quadrants() -> None:
    profiles = build_segment_profiles(_clustered()).set_index("Cluster")
    assert profiles.loc[0, "Persona"] == PERSONA_DEFAULTS["champions"]
    assert profiles.loc[1, "Persona"] == PERSONA_DEFAULTS["promising"]
    assert profiles.loc[2, "Persona"] == PERSONA_DEFAULTS["at_risk"]
    assert profiles.loc[3, "Persona"] == PERSONA_DEFAULTS["lost"]


def test_sizes_and_shares() -> None:
    profiles = build_segment_profiles(_clustered())
    assert profiles["Size"].sum() == 40
    assert profiles["Share"].sum() == pytest.approx(100.0, abs=0.5)


def test_custom_persona_names() -> None:
    profiles = build_segment_profiles(
        _clustered(), persona_names={"champions": "VIP"}
    ).set_index("Cluster")
    assert profiles.loc[0, "Persona"] == "VIP"


def test_missing_column_raises() -> None:
    df = _clustered().drop(columns=["Monetary"])
    with pytest.raises(ValueError, match="Missing required columns"):
        build_segment_profiles(df)
