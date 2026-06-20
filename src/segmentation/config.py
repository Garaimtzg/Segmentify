"""Load and validate the project configuration from ``config/config.yaml``.

No paths are hardcoded in the rest of the code base: every module receives a
:class:`Config` instance built from the YAML file. Paths are resolved relative
to the project root (the directory that contains ``config/``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/segmentation/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class Config:
    """Typed view over ``config.yaml``."""

    seed: int
    paths: Paths
    dataset: dict[str, Any]
    artifacts: dict[str, str]
    etl: dict[str, Any]
    rfm: dict[str, Any]
    model: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # --- Convenience accessors for the concrete artifact paths --------------
    @property
    def raw_dataset_path(self) -> Path:
        return self.paths.raw_dir / self.dataset["raw_filename"]

    @property
    def transactions_clean_path(self) -> Path:
        return self.paths.interim_dir / self.artifacts["transactions_clean"]

    @property
    def rfm_path(self) -> Path:
        return self.paths.processed_dir / self.artifacts["rfm"]

    @property
    def rfm_clustered_path(self) -> Path:
        return self.paths.processed_dir / self.artifacts["rfm_clustered"]

    @property
    def segment_profiles_path(self) -> Path:
        return self.paths.reports_dir / self.artifacts["segment_profiles"]


def _resolve(root: Path, value: str) -> Path:
    """Resolve a possibly-relative config path against the project root."""
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read ``config.yaml`` and return a validated :class:`Config`.

    Raises:
        FileNotFoundError: if the config file does not exist.
        KeyError: if a required top-level section is missing.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    for section in ("seed", "paths", "dataset", "artifacts", "etl", "rfm", "model"):
        if section not in raw:
            raise KeyError(f"Missing required config section: '{section}'")

    root = config_path.resolve().parent.parent
    p = raw["paths"]
    paths = Paths(
        raw_dir=_resolve(root, p["raw_dir"]),
        interim_dir=_resolve(root, p["interim_dir"]),
        processed_dir=_resolve(root, p["processed_dir"]),
        reports_dir=_resolve(root, p["reports_dir"]),
        figures_dir=_resolve(root, p["figures_dir"]),
    )

    return Config(
        seed=int(raw["seed"]),
        paths=paths,
        dataset=raw["dataset"],
        artifacts=raw["artifacts"],
        etl=raw["etl"],
        rfm=raw["rfm"],
        model=raw["model"],
        raw=raw,
    )
