"""Load stage: persist and read back intermediate/processed artifacts.

Centralizes parquet I/O so the rest of the package stays free of file handling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` to ``path`` as parquet, creating parent dirs as needed.

    Returns:
        The resolved output path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def save_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Write ``df`` to ``path`` as CSV, creating parent dirs as needed.

    Returns:
        The resolved output path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=index)
    return out_path


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file produced by the pipeline.

    Raises:
        FileNotFoundError: if the artifact does not exist yet.
    """
    in_path = Path(path)
    if not in_path.is_file():
        raise FileNotFoundError(
            f"Artifact not found at '{in_path}'. Run the corresponding "
            "pipeline stage first (see `make pipeline`)."
        )
    return pd.read_parquet(in_path)
