"""RFM feature engineering.

From cleaned transactions, compute per-customer **Recency**, **Frequency** and
**Monetary** features. Pure and testable (no I/O).

Definitions (see brief §5, Milestone 3):

* **Recency:** days from the customer's last purchase to the ``snapshot_date``.
* **Frequency:** number of distinct invoices.
* **Monetary:** sum of ``TotalPrice``.

If ``snapshot_date`` is ``None`` it defaults to ``max(InvoiceDate) + 1 day``.
Recency is computed at day granularity (the time-of-day component of invoices is
ignored), so results are clean integer day counts.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"]


def resolve_snapshot_date(
    transactions: pd.DataFrame,
    snapshot_date: str | pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Return the snapshot date used for Recency.

    Defaults to the latest invoice date plus one day when not provided.
    Normalized to midnight so Recency is measured in whole days.
    """
    if snapshot_date is not None:
        return pd.Timestamp(snapshot_date).normalize()
    latest = pd.to_datetime(transactions["InvoiceDate"]).max().normalize()
    return latest + pd.Timedelta(days=1)


def compute_rfm(
    transactions: pd.DataFrame,
    snapshot_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute the per-customer RFM matrix.

    Args:
        transactions: Cleaned transactions with ``CustomerID``, ``InvoiceNo``,
            ``InvoiceDate`` and ``TotalPrice`` columns.
        snapshot_date: Reference date for Recency. If ``None``, uses
            ``max(InvoiceDate) + 1 day``.

    Returns:
        A DataFrame with one row per customer and columns
        ``CustomerID``, ``Recency``, ``Frequency``, ``Monetary``.

    Raises:
        ValueError: if a required column is missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in transactions.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = transactions.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    snapshot = resolve_snapshot_date(df, snapshot_date)

    grouped = df.groupby("CustomerID")
    last_purchase = grouped["InvoiceDate"].max().dt.normalize()
    rfm = pd.DataFrame(
        {
            "Recency": (snapshot - last_purchase).dt.days,
            "Frequency": grouped["InvoiceNo"].nunique(),
            "Monetary": grouped["TotalPrice"].sum(),
        }
    )

    return rfm.reset_index()
