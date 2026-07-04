"""Shared pytest fixtures: a tiny synthetic dataset mirroring Online Retail.

The fixture is hand-crafted so RFM values can be checked by hand (see
``test_rfm.py``). It deliberately includes rows that must be dropped by the ETL:
a null ``CustomerID``, a cancellation (``InvoiceNo`` starting with ``C``), a
negative ``Quantity`` and a zero ``UnitPrice``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# --- Valid rows -----------------------------------------------------------
# Customer 1001: invoices {536365, 536366} -> Frequency = 2
#   536365 has two line items (15.0 + 3.0), 536366 has one (10.0)
#   Monetary = 28.0 ; last purchase = 2011-01-10
# Customer 1002: invoice {536400} -> Frequency = 1
#   Monetary = 100.0 ; last purchase = 2011-02-01
#
# With snapshot_date = max(InvoiceDate) + 1 day = 2011-02-02:
#   Recency(1001) = 2011-02-02 - 2011-01-10 = 23 days
#   Recency(1002) = 2011-02-02 - 2011-02-01 =  1 day
_RAW_ROWS = [
    # InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
    ("536365", "85123A", "WHITE HANGING HEART", 6, "2011-01-01 08:26:00", 2.5, 1001.0, "United Kingdom"),
    ("536365", "71053", "WHITE METAL LANTERN", 3, "2011-01-01 08:26:00", 1.0, 1001.0, "United Kingdom"),
    ("536366", "84406B", "CREAM CUPID HEARTS", 2, "2011-01-10 09:01:00", 5.0, 1001.0, "United Kingdom"),
    ("536400", "22423", "REGENCY CAKESTAND", 1, "2011-02-01 12:00:00", 100.0, 1002.0, "France"),
    # --- rows that MUST be dropped ---
    ("536500", "21730", "GLASS STAR", 5, "2011-01-15 10:00:00", 3.0, np.nan, "United Kingdom"),   # null CustomerID
    ("C536501", "85123A", "WHITE HANGING HEART", -2, "2011-01-16 11:00:00", 2.5, 1001.0, "United Kingdom"),  # cancellation
    ("536502", "22423", "REGENCY CAKESTAND", -5, "2011-01-17 11:00:00", 4.0, 1002.0, "France"),   # negative qty
    ("536503", "84406B", "CREAM CUPID HEARTS", 5, "2011-01-18 11:00:00", 0.0, 1001.0, "United Kingdom"),  # zero price
]

_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


@pytest.fixture
def raw_transactions() -> pd.DataFrame:
    """Raw (uncleaned) synthetic transactions, including invalid rows."""
    df = pd.DataFrame(_RAW_ROWS, columns=_COLUMNS)
    # InvoiceDate stays as string to exercise datetime parsing in transform.
    return df


@pytest.fixture
def clean_transactions_df(raw_transactions: pd.DataFrame) -> pd.DataFrame:
    """Cleaned synthetic transactions (4 valid rows)."""
    from segmentation.etl.transform import clean_transactions

    return clean_transactions(raw_transactions)
