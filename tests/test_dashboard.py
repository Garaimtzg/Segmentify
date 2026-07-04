"""Tests for the dashboard data-preparation helper (Milestone 9)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from segmentation.dashboard.app import prepare_dashboard_data


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    clustered = pd.DataFrame(
        {
            "CustomerID": range(1, 21),
            "Recency": list(range(1, 21)),
            "Frequency": list(range(1, 21)),
            "Monetary": [100.0 * i for i in range(1, 21)],
            "Cluster": [0, 1] * 10,
        }
    )
    profiles = pd.DataFrame(
        {
            "Cluster": [0, 1],
            "Persona": ["Campeones / VIP", "Dormidos / Perdidos"],
            "Size": [10, 10],
            "Share": [50.0, 50.0],
            "Recency": [5.0, 15.0],
            "Frequency": [5.0, 15.0],
            "Monetary": [500.0, 1500.0],
        }
    )
    clustered_path = tmp_path / "rfm_clustered.parquet"
    profiles_path = tmp_path / "segment_profiles.csv"
    clustered.to_parquet(clustered_path, index=False)
    profiles.to_csv(profiles_path, index=False)
    return clustered_path, profiles_path


def test_returns_none_when_missing(tmp_path: Path) -> None:
    assert prepare_dashboard_data(
        tmp_path / "nope.parquet", tmp_path / "nope.csv"
    ) is None


def test_attaches_persona_and_pca(tmp_path: Path) -> None:
    clustered_path, profiles_path = _write_artifacts(tmp_path)
    result = prepare_dashboard_data(clustered_path, profiles_path)
    assert result is not None
    clustered, profiles, ev = result

    assert {"Persona", "PC1", "PC2"}.issubset(clustered.columns)
    assert clustered["Persona"].notna().all()
    assert set(clustered["Persona"].unique()) == {
        "Campeones / VIP",
        "Dormidos / Perdidos",
    }
    assert len(ev) == 2
    assert all(0.0 <= v <= 1.0 for v in ev)
