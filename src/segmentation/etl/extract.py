"""Extract stage: read the raw transactional dataset.

This module performs I/O only. If the dataset is missing it stops with a clear
message asking for it — it never fabricates data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns we expect in the UCI Online Retail dataset.
EXPECTED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def read_raw_transactions(path: str | Path, sheet_name: int | str = 0) -> pd.DataFrame:
    """Read the raw Online Retail dataset from an ``.xlsx`` file.

    Args:
        path: Path to the raw dataset (e.g. ``data/raw/Online Retail.xlsx``).
        sheet_name: Sheet index or name to read.

    Returns:
        The raw transactions as a :class:`pandas.DataFrame` (uncleaned).

    Raises:
        FileNotFoundError: if the dataset is not present. The message explains
            where to place it instead of generating synthetic data.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at '{dataset_path}'.\n"
            "Download the UCI 'Online Retail' dataset (.xlsx) and place it "
            "there. The expected filename is configured in "
            "config/config.yaml -> dataset.raw_filename. "
            "Data is never fabricated."
        )

    return pd.read_excel(dataset_path, sheet_name=sheet_name)
