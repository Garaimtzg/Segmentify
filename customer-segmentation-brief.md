# Customer Segmentation (RFM + Clustering) — Brief para Claude Code

> Documento maestro del proyecto. Pensado para entregárselo a Claude Code como
> especificación: define el objetivo, el stack, la estructura del repo y una
> secuencia de *milestones* con criterios de aceptación. Trabájalo milestone a
> milestone, no todo de golpe.

---

## 0. Objetivo

Construir un proyecto de **segmentación de clientes de nivel ingenieril** sobre
datos transaccionales reales. El entregable NO es un notebook suelto, sino un
**paquete Python modular** con pipeline ETL reproducible, ingeniería de
características RFM, modelado no supervisado (K-Means), reducción de
dimensionalidad (PCA), visualizaciones y un análisis de negocio documentado.

El proyecto debe poder levantarse en Linux con un entorno virtual y ejecutarse
de principio a fin con un solo comando.

---

## 1. Stack tecnológico

- **Lenguaje:** Python 3.11+
- **Datos:** pandas, numpy, openpyxl (el dataset UCI viene en `.xlsx`)
- **ML:** scikit-learn (KMeans, StandardScaler, PCA, silhouette_score)
- **Visualización:** matplotlib, seaborn
- **Dashboard:** streamlit (app ligera para explorar segmentos)
- **Config:** PyYAML (config externa, nada hardcodeado)
- **CLI:** argparse o typer (subcomandos)
- **Tests:** pytest
- **Calidad:** ruff (lint + format), opcionalmente mypy
- **Orquestación local:** Makefile con targets

Pinea versiones en `requirements.txt`. Fija `random_state` en todo lo
estocástico para reproducibilidad.

---

## 2. Estructura del repositorio

```
customer-segmentation/
├── README.md
├── CLAUDE.md                 # convenciones para el agente
├── requirements.txt
├── pyproject.toml            # config de ruff/pytest (opcional pero recomendado)
├── Makefile
├── .gitignore                # ignora data/raw, data/interim, venv, __pycache__
├── config/
│   └── config.yaml           # rutas, snapshot_date, k_range, seed, etc.
├── data/
│   ├── raw/                  # dataset original descargado (no se commitea)
│   ├── interim/              # datos limpios intermedios
│   └── processed/            # matriz RFM y resultados con etiquetas
├── src/
│   └── segmentation/
│       ├── __init__.py
│       ├── config.py         # carga y valida config.yaml (dataclass)
│       ├── etl/
│       │   ├── extract.py    # lee el dataset crudo
│       │   ├── transform.py  # limpieza, devoluciones, fechas, nulos
│       │   └── load.py       # persiste a interim/processed
│       ├── features/
│       │   └── rfm.py        # cálculo de Recency, Frequency, Monetary
│       ├── models/
│       │   ├── selection.py  # Elbow + silhouette para elegir K
│       │   └── clustering.py # escalado + KMeans + asignación de etiquetas
│       ├── viz/
│       │   └── plots.py      # elbow, silhouette, scatter PCA, perfiles RFM
│       ├── analysis/
│       │   └── personas.py   # mapea clústeres a nombres de negocio
│       ├── cli.py            # punto de entrada: etl|features|train|viz|all
│       └── dashboard/
│           └── app.py        # app Streamlit (lee data/processed)
├── notebooks/
│   └── 01_exploration.ipynb  # EDA inicial (opcional, no es el entregable)
├── reports/
│   └── figures/              # PNGs generados por el pipeline
└── tests/
    ├── conftest.py           # fixtures con un mini-dataset sintético
    ├── test_transform.py
    ├── test_rfm.py
    └── test_clustering.py
```

---

## 3. Setup del entorno (Linux) — debe quedar documentado en el README

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

El `Makefile` debe exponer al menos:

```
make install     # crea venv e instala deps
make lint        # ruff check
make test        # pytest
make etl         # ejecuta solo el ETL
make pipeline    # ejecuta el flujo completo (etl -> features -> train -> viz)
make dashboard   # levanta la app Streamlit en local
make clean       # borra interim/processed y figuras
```

---

## 4. Dataset

**Online Retail Dataset** (UCI Machine Learning Repository). Es un fichero
`.xlsx` con cientos de miles de registros de facturas de un retailer online del
Reino Unido. Columnas típicas: `InvoiceNo`, `StockCode`, `Description`,
`Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

Notas importantes para el ETL:
- Las facturas que empiezan por `C` en `InvoiceNo` son **devoluciones** → filtrar.
- Hay filas con `CustomerID` nulo → no sirven para segmentar → eliminar.
- `Quantity` y `UnitPrice` pueden ser ≤ 0 → eliminar.
- `InvoiceDate` debe parsearse a datetime.
- Crear columna derivada `TotalPrice = Quantity * UnitPrice`.

La descarga del dataset puede requerir intervención manual (Kaggle pide login).
Claude Code debe: intentar descargarlo desde la fuente; si no puede, **parar y
pedirme que lo coloque manualmente en `data/raw/`**, dejando documentado el
nombre de fichero esperado. No inventar datos.

---

## 5. Milestones (ejecutar en orden)

### Milestone 1 — Andamiaje del proyecto
Crear la estructura de carpetas, `requirements.txt`, `pyproject.toml`,
`Makefile`, `.gitignore`, `config/config.yaml`, `CLAUDE.md` y los `__init__.py`.
El paquete debe ser importable y `make install` debe funcionar.

**Aceptación:** `python -c "import segmentation"` sin error; `make lint` pasa.

### Milestone 2 — Pipeline ETL (Fase 2)
Implementar `extract.py` (lectura), `transform.py` (limpieza según §4) y
`load.py` (persistencia a `data/interim/transactions_clean.parquet`).

**Aceptación:** `make etl` produce el fichero limpio; los tests de
`test_transform.py` verifican que se eliminan nulos, devoluciones (`C…`) y
cantidades/precios no positivos, y que existe `TotalPrice`.

### Milestone 3 — Ingeniería de características RFM (Fase 3)
En `rfm.py`, a partir de las transacciones limpias y un `snapshot_date`
(= fecha máxima de factura + 1 día, configurable), calcular por cliente:
- **Recency:** días desde la última compra hasta el snapshot.
- **Frequency:** nº de facturas distintas.
- **Monetary:** suma de `TotalPrice`.

Guardar en `data/processed/rfm.parquet`.

**Aceptación:** `test_rfm.py` comprueba los valores RFM sobre un cliente del
fixture sintético con resultados calculados a mano.

### Milestone 4 — Selección de K (Fase 4)
En `selection.py`: escalar la matriz RFM (recomendado: `log1p` para corregir el
sesgo de Frequency/Monetary, luego `StandardScaler`) y barrer un rango de K
(p.ej. 2–10) calculando **inercia (Elbow)** y **silhouette score**. Exportar las
métricas a `reports/figures/elbow.png` y `silhouette.png`, y dejar el K elegido
en el log y/o en `config.yaml`.

**Aceptación:** se generan ambas figuras; la función devuelve un K razonable
(normalmente 3–5) con justificación numérica.

### Milestone 5 — Clustering (Fase 4)
En `clustering.py`: entrenar K-Means con el K elegido y `random_state` fijo,
asignar etiqueta a cada cliente y persistir
`data/processed/rfm_clustered.parquet`.

**Aceptación:** `test_clustering.py` verifica que cada cliente recibe una
etiqueta y que el nº de clústeres coincide con K.

### Milestone 6 — PCA y visualización (Fase 5)
En `plots.py`: aplicar PCA (2D y 3D) sobre las features escaladas y generar
scatter coloreado por clúster, más un gráfico de perfiles (media de R, F y M por
clúster). Guardar todo en `reports/figures/`.

**Aceptación:** existen `pca_2d.png` (mínimo) y un gráfico de perfiles RFM;
`make pipeline` los regenera sin pasos manuales.

### Milestone 7 — Análisis de negocio (Fase 6)
En `personas.py`: a partir de la media RFM por clúster, asignar nombres
interpretables (p.ej. *Campeones/VIP*, *Leales*, *En riesgo*, *Dormidos/Perdidos*)
y exportar una tabla resumen (`reports/segment_profiles.csv`) con tamaño del
segmento y RFM medio.

**Aceptación:** la tabla relaciona cada clúster con su persona y sus métricas.

### Milestone 8 — CLI y orquestación
`cli.py` con subcomandos `etl`, `features`, `train`, `viz`, `analysis` y `all`.
`make pipeline` debe ejecutar `all` de extremo a extremo.

**Aceptación:** `python -m segmentation.cli all` corre el flujo completo desde
`data/raw` hasta las figuras y la tabla de segmentos.

### Milestone 9 — Dashboard ligero con Streamlit
En `dashboard/app.py`, una app que **consume los artefactos ya generados**
(`data/processed/rfm_clustered.parquet` y `reports/segment_profiles.csv`) — no
reentrena nada, solo explora resultados. Debe incluir:
- Tabla de perfiles de segmento (tamaño + RFM medio + nombre de persona).
- Scatter PCA 2D interactivo coloreado por clúster.
- Filtro por segmento/persona en la barra lateral.
- KPIs arriba: nº de clientes, nº de segmentos, gasto medio.

Si los parquet no existen, mostrar un aviso pidiendo correr `make pipeline`
primero (no debe petar). Documentar el arranque en el README:
`streamlit run src/segmentation/dashboard/app.py` (o `make dashboard`).

**Aceptación:** `make dashboard` levanta la app sin errores con los datos ya
procesados y permite filtrar por segmento.

### Milestone 10 — Documentación y GitHub (Fase 6)
README completo: descripción, motivación, cómo conseguir el dataset, setup del
venv en Linux, cómo ejecutar, cómo levantar el dashboard, explicación de cada
segmento (con una figura incrustada) e interpretación de negocio. Inicializar
git, `.gitignore` correcto (que NO suba `data/raw`), commits con mensajes
claros.

**Aceptación:** el README permite a un tercero clonar, instalar y reproducir
los resultados sin ayuda.

---

## 6. Contenido sugerido para `CLAUDE.md`

```markdown
# Convenciones del proyecto
- Python 3.11+, código en `src/segmentation/`, tests en `tests/`.
- Nada de rutas hardcodeadas: todo desde `config/config.yaml`.
- Fija `random_state` (seed en config) en KMeans/PCA y donde aplique.
- Cada módulo expone funciones puras y testeables; el I/O vive en etl/load.
- Antes de cerrar un milestone: `make lint` y `make test` deben pasar.
- No commitear datos crudos ni el venv.
- Si falta el dataset, parar y pedirlo; nunca generar datos falsos.
```

---

## 7. Prompt inicial sugerido para Claude Code

> Lee `customer-segmentation-brief.md`. Implementa el proyecto siguiendo los
> milestones en orden. Empieza por el Milestone 1 (andamiaje): crea la
> estructura de carpetas, `requirements.txt`, `pyproject.toml`, `Makefile`,
> `.gitignore`, `config/config.yaml` y `CLAUDE.md`. Cuando termines, ejecuta
> `make lint`, muéstrame el resultado y espera mi confirmación antes de pasar al
> Milestone 2.

Avanzar milestone a milestone (pidiendo confirmación entre cada uno) te da
control y mantiene los commits limpios, que es justo lo que un revisor de un
máster querrá ver en el historial de GitHub.
