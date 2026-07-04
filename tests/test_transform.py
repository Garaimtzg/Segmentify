"""Tests for the ETL transform stage (Milestone 2)."""

from __future__ import annotations

import pandas as pd
import pytest

from segmentation.etl.transform import clean_transactions


def test_drops_null_customer(clean_transactions_df: pd.DataFrame) -> None:
    assert clean_transactions_df["CustomerID"].notna().all()


def test_drops_cancellations(clean_transactions_df: pd.DataFrame) -> None:
    assert not clean_transactions_df["InvoiceNo"].str.upper().str.startswith("C").any()


def test_drops_non_positive_quantity_and_price(
    clean_transactions_df: pd.DataFrame,
) -> None:
    assert (clean_transactions_df["Quantity"] > 0).all()
    assert (clean_transactions_df["UnitPrice"] > 0).all()


def test_total_price_column(clean_transactions_df: pd.DataFrame) -> None:
    expected = clean_transactions_df["Quantity"] * clean_transactions_df["UnitPrice"]
    assert "TotalPrice" in clean_transactions_df.columns
    pd.testing.assert_series_equal(
        clean_transactions_df["TotalPrice"], expected, check_names=False
    )


def test_invoice_date_is_datetime(clean_transactions_df: pd.DataFrame) -> None:
    assert pd.api.types.is_datetime64_any_dtype(clean_transactions_df["InvoiceDate"])


def test_customer_id_is_integer(clean_transactions_df: pd.DataFrame) -> None:
    assert pd.api.types.is_integer_dtype(clean_transactions_df["CustomerID"])


def test_keeps_only_valid_rows(clean_transactions_df: pd.DataFrame) -> None:
    # 3 valid rows for customer 1001 + 1 for customer 1002.
    assert len(clean_transactions_df) == 4
    assert set(clean_transactions_df["CustomerID"]) == {1001, 1002}


def test_missing_required_column_raises(raw_transactions: pd.DataFrame) -> None:
    broken = raw_transactions.drop(columns=["CustomerID"])
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_transactions(broken)
