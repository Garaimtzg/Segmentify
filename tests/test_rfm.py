"""Tests for RFM feature engineering (Milestone 3).

Values are hand-computed from the synthetic fixture (see ``conftest.py``).
With snapshot_date = max(InvoiceDate) + 1 day = 2011-02-02:

* Customer 1001: Recency 23, Frequency 2, Monetary 28.0
* Customer 1002: Recency  1, Frequency 1, Monetary 100.0
"""

from __future__ import annotations

import pandas as pd

from segmentation.features.rfm import compute_rfm, resolve_snapshot_date


def test_default_snapshot_is_max_date_plus_one_day(
    clean_transactions_df: pd.DataFrame,
) -> None:
    snapshot = resolve_snapshot_date(clean_transactions_df)
    assert snapshot == pd.Timestamp("2011-02-02")


def test_rfm_values_match_hand_computation(
    clean_transactions_df: pd.DataFrame,
) -> None:
    rfm = compute_rfm(clean_transactions_df).set_index("CustomerID")

    assert set(rfm.columns) == {"Recency", "Frequency", "Monetary"}

    assert rfm.loc[1001, "Recency"] == 23
    assert rfm.loc[1001, "Frequency"] == 2
    assert rfm.loc[1001, "Monetary"] == 28.0

    assert rfm.loc[1002, "Recency"] == 1
    assert rfm.loc[1002, "Frequency"] == 1
    assert rfm.loc[1002, "Monetary"] == 100.0


def test_one_row_per_customer(clean_transactions_df: pd.DataFrame) -> None:
    rfm = compute_rfm(clean_transactions_df)
    assert len(rfm) == clean_transactions_df["CustomerID"].nunique()
    assert rfm["CustomerID"].is_unique


def test_explicit_snapshot_date(clean_transactions_df: pd.DataFrame) -> None:
    # One day later than the default -> every Recency shifts by +1.
    rfm = compute_rfm(clean_transactions_df, snapshot_date="2011-02-03")
    rfm = rfm.set_index("CustomerID")
    assert rfm.loc[1001, "Recency"] == 24
    assert rfm.loc[1002, "Recency"] == 2


def test_missing_required_column_raises(
    clean_transactions_df: pd.DataFrame,
) -> None:
    import pytest

    broken = clean_transactions_df.drop(columns=["TotalPrice"])
    with pytest.raises(ValueError, match="Missing required columns"):
        compute_rfm(broken)
