# Convenciones del proyecto
- Python 3.11+, código en `src/segmentation/`, tests en `tests/`.
- Nada de rutas hardcodeadas: todo desde `config/config.yaml`.
- Fija `random_state` (seed en config) en KMeans/PCA y donde aplique.
- Cada módulo expone funciones puras y testeables; el I/O vive en etl/load.
- Antes de cerrar un milestone: `make lint` y `make test` deben pasar.
- No commitear datos crudos ni el venv.
- Si falta el dataset, parar y pedirlo; nunca generar datos falsos.