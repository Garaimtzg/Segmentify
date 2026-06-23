"""Transform stage: clean raw transactions into an analysis-ready table.

Pure, testable transformations (no I/O). Cleaning rules (see brief §4):

* Parse ``InvoiceDate`` to datetime.
* Drop rows with a null ``CustomerID`` (cannot be segmented).
* Drop returns / cancellations (``InvoiceNo`` starting with ``C``).
* Drop rows with non-positive ``Quantity`` or ``UnitPrice``.
* Add the derived column ``TotalPrice = Quantity * UnitPrice``.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
]


def clean_transactions(
    df: pd.DataFrame,
    *,
    cancellation_prefix: str = "C",
    min_quantity: float = 0,
    min_unit_price: float = 0,
) -> pd.DataFrame:
    """Clean raw transactions according to the project's ETL rules.

    Args:
        df: Raw transactions (as returned by ``read_raw_transactions``).
        cancellation_prefix: ``InvoiceNo`` prefix marking returns to drop.
        min_quantity: rows with ``Quantity`` <= this value are dropped.
        min_unit_price: rows with ``UnitPrice`` <= this value are dropped.

    Returns:
        A cleaned copy with a ``TotalPrice`` column, integer ``CustomerID`` and
        datetime ``InvoiceDate``. The index is reset.

    Raises:
        ValueError: if a required column is missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # Normalize types.
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"])
    out["InvoiceNo"] = out["InvoiceNo"].astype(str).str.strip()
    # StockCode mixes strings ("85123A") and ints (22423); make it consistently
    # string so downstream serialization (parquet) is well-typed.
    if "StockCode" in out.columns:
        out["StockCode"] = out["StockCode"].astype(str).str.strip()

    # Drop rows we cannot use.
    out = out[out["CustomerID"].notna()]
    prefix = cancellation_prefix.upper()
    out = out[~out["InvoiceNo"].str.upper().str.startswith(prefix)]
    out = out[out["Quantity"] > min_quantity]
    out = out[out["UnitPrice"] > min_unit_price]

    # Derived column + tidy dtypes.
    out["TotalPrice"] = out["Quantity"] * out["UnitPrice"]
    out["CustomerID"] = out["CustomerID"].astype("int64")

    return out.reset_index(drop=True)
