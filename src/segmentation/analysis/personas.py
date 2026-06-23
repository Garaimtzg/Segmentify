"""Business analysis: map clusters to interpretable personas.

From the per-cluster mean RFM, assign a business persona using a transparent,
data-driven rule (no hardcoded cluster IDs): split clusters by whether they are
*recent* (Recency below the median of cluster means) and *valuable* (Monetary
above the median of cluster means). This yields the classic RFM quadrants:

* recent & valuable      -> Champions / VIP
* recent & low value     -> Promising / New
* not recent & valuable  -> At risk (were valuable, slipping away)
* not recent & low value  -> Dormant / Lost

Persona display names are configurable (see ``config.yaml`` -> ``personas``).
Pure and testable: no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

RFM_FEATURES = ["Recency", "Frequency", "Monetary"]

# Default (Spanish) persona names, keyed by quadrant.
PERSONA_DEFAULTS = {
    "champions": "Campeones / VIP",
    "promising": "Prometedores",
    "at_risk": "En riesgo",
    "lost": "Dormidos / Perdidos",
}


def _persona_for(
    recency: float,
    monetary: float,
    recency_median: float,
    monetary_median: float,
    names: dict[str, str],
) -> str:
    recent = recency <= recency_median
    valuable = monetary >= monetary_median
    if recent and valuable:
        return names["champions"]
    if recent and not valuable:
        return names["promising"]
    if not recent and valuable:
        return names["at_risk"]
    return names["lost"]


def build_segment_profiles(
    clustered: pd.DataFrame,
    *,
    persona_names: dict[str, str] | None = None,
    cluster_col: str = "Cluster",
    features: Sequence[str] = RFM_FEATURES,
) -> pd.DataFrame:
    """Summarize each cluster and attach a persona name.

    Args:
        clustered: RFM table with a cluster-label column.
        persona_names: Optional overrides for the default persona names.
        cluster_col: Name of the cluster-label column.
        features: RFM feature columns (Recency, Frequency, Monetary).

    Returns:
        One row per cluster with columns: ``Cluster``, ``Persona``, ``Size``,
        ``Share`` (% of customers), and the mean of each RFM feature. Sorted by
        Monetary (most valuable first).

    Raises:
        ValueError: if a required column is missing.
    """
    required = [cluster_col, *features]
    missing = [c for c in required if c not in clustered.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    recency_col, frequency_col, monetary_col = features

    grouped = clustered.groupby(cluster_col)
    profiles = grouped[list(features)].mean()
    profiles["Size"] = grouped.size()
    profiles["Share"] = profiles["Size"] / profiles["Size"].sum() * 100.0

    names = {**PERSONA_DEFAULTS, **(persona_names or {})}
    recency_median = profiles[recency_col].median()
    monetary_median = profiles[monetary_col].median()
    profiles["Persona"] = [
        _persona_for(
            row[recency_col], row[monetary_col],
            recency_median, monetary_median, names,
        )
        for _, row in profiles.iterrows()
    ]

    profiles = profiles.reset_index()
    profiles = profiles.round(
        {recency_col: 1, frequency_col: 2, monetary_col: 2, "Share": 1}
    )
    profiles["Size"] = profiles["Size"].astype(int)

    ordered_cols = [
        cluster_col, "Persona", "Size", "Share",
        recency_col, frequency_col, monetary_col,
    ]
    profiles = profiles[ordered_cols]
    return profiles.sort_values(monetary_col, ascending=False).reset_index(drop=True)
