"""Streamlit dashboard to explore the customer segments.

This app *consumes* the artifacts produced by the pipeline
(``data/processed/rfm_clustered.parquet`` and ``reports/segment_profiles.csv``)
— it never retrains anything. PCA is recomputed only for display (deterministic).

Run it with::

    streamlit run src/segmentation/dashboard/app.py
    # or: make dashboard

If the artifacts are missing, the app shows a friendly notice (it does not crash).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from segmentation.config import load_config
from segmentation.models.selection import scale_features
from segmentation.viz.plots import project_pca


def prepare_dashboard_data(
    clustered_path: str | Path,
    profiles_path: str | Path,
    *,
    log1p: bool = True,
    standardize: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[float]] | None:
    """Load artifacts, attach persona names and 2D PCA coordinates.

    Returns ``(clustered, profiles, explained_variance)`` or ``None`` if either
    artifact is missing. Pure (no Streamlit calls), so it is unit-testable.
    """
    clustered_path = Path(clustered_path)
    profiles_path = Path(profiles_path)
    if not clustered_path.is_file() or not profiles_path.is_file():
        return None

    clustered = pd.read_parquet(clustered_path)
    profiles = pd.read_csv(profiles_path)

    persona_map = dict(zip(profiles["Cluster"], profiles["Persona"], strict=False))
    clustered["Persona"] = clustered["Cluster"].map(persona_map)

    X = scale_features(clustered, log1p=log1p, standardize=standardize)
    coords, ev = project_pca(X, n_components=2, random_state=random_state)
    clustered["PC1"] = coords[:, 0]
    clustered["PC2"] = coords[:, 1]

    return clustered, profiles, [float(v) for v in ev]


@st.cache_data(show_spinner=False)
def _load_cached() -> tuple[pd.DataFrame, pd.DataFrame, list[float]] | None:
    cfg = load_config()
    return prepare_dashboard_data(
        cfg.rfm_clustered_path,
        cfg.segment_profiles_path,
        log1p=cfg.model.get("log1p", True),
        standardize=cfg.model.get("standardize", True),
        random_state=cfg.model["random_state"],
    )


def main() -> None:
    st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")
    st.title("🛍️ Customer Segmentation Dashboard")
    st.caption("RFM + K-Means · explora los segmentos generados por el pipeline")

    data = _load_cached()
    if data is None:
        st.warning(
            "No se encontraron los artefactos procesados. Ejecuta primero el "
            "pipeline:\n\n```bash\nmake pipeline\n```",
            icon="⚠️",
        )
        st.stop()

    clustered, profiles, ev = data

    # --- Sidebar filter ---------------------------------------------------
    personas = (
        profiles.sort_values("Monetary", ascending=False)["Persona"].tolist()
    )
    selected = st.sidebar.multiselect(
        "Filtrar por segmento", options=personas, default=personas
    )
    if not selected:
        st.info("Selecciona al menos un segmento en la barra lateral.")
        st.stop()

    view = clustered[clustered["Persona"].isin(selected)]

    # --- KPIs -------------------------------------------------------------
    k1, k2, k3 = st.columns(3)
    k1.metric("Clientes", f"{len(view):,}")
    k2.metric("Segmentos", int(view["Cluster"].nunique()))
    k3.metric("Gasto medio", f"£{view['Monetary'].mean():,.0f}")

    st.divider()

    # --- Scatter + profiles table ----------------------------------------
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Segmentos en espacio PCA (2D)")
        st.scatter_chart(
            view, x="PC1", y="PC2", color="Persona", height=460
        )
        st.caption(
            f"Varianza explicada: PC1 {ev[0] * 100:.1f}% · PC2 {ev[1] * 100:.1f}%"
        )
    with right:
        st.subheader("Perfiles de segmento")
        prof_view = profiles[profiles["Persona"].isin(selected)]
        st.dataframe(prof_view, hide_index=True, use_container_width=True)
        st.download_button(
            "Descargar clientes filtrados (CSV)",
            data=view.drop(columns=["PC1", "PC2"]).to_csv(index=False),
            file_name="customers_filtered.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
