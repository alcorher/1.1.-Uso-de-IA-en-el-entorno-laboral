# Guía de ejecución paso a paso

Esta guía parte de un clon del repositorio y utiliza `uv` para resolver dependencias y ejecutar Python dentro del entorno del proyecto.

## Prerrequisitos e instalación de `uv`

El proyecto requiere Python 3.11 o superior. Instala `uv` con el instalador oficial de Astral:

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux y macOS:**

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Abre una terminal nueva y verifica que esté disponible:

```text
uv --version
uv 0.12.21
```

La salida anterior corresponde al entorno de desarrollo; la versión instalada puede variar. Los instaladores se obtienen del sitio oficial de Astral; aplica las políticas de seguridad de tu entorno antes de ejecutar instaladores remotos.

## Preparación del entorno virtual

Desde la carpeta raíz del repositorio:

```sh
uv sync
```

`uv sync` lee `pyproject.toml` y `uv.lock`, crea `.venv` si no existe e instala las dependencias del proyecto y del grupo de desarrollo. El lockfile registra las versiones resueltas para que las instalaciones sean reproducibles. No hace falta activar el entorno virtual manualmente: `uv run` ejecuta los comandos usando el entorno del proyecto.

Dependencias declaradas: `pandas`, `scikit-learn` y `joblib`; para las pruebas, `pytest`.

## Ejecución de los módulos

Ejecuta los comandos desde la raíz del proyecto y en el orden indicado para generar datos y modelo antes de inferir.

### 1. Generar el dataset

```sh
uv run python src/generate_dataset.py
```

Usa plantillas y una semilla fija (`42`) para crear 600 incidencias. No lee datos de entrada; crea o sobrescribe `data/tickets_dataset.csv`.

Salida real de la ejecución:

```text
Dataset generado: 600 incidencias en …\data\tickets_dataset.csv
Distribución por categoría:
category
bug            188
docs            90
feature        140
performance     91
security        91
```

El prefijo de la ruta depende de dónde se haya clonado el repositorio. La distribución es la obtenida con el generador y semilla actuales.

### 2. Entrenar y evaluar

```sh
uv run python src/train.py
```

Lee `data/tickets_dataset.csv`, combina `title` y `description`, limpia el texto, separa entrenamiento y prueba de forma estratificada, ajusta el pipeline y muestra el reporte y la matriz de confusión. Crea o actualiza `models/triage_pipeline.joblib` y `docs/metrics.json`.

Salida real de la ejecución:

```text
Reporte de clasificación:
              precision    recall  f1-score   support

         bug       0.95      0.92      0.93        38
        docs       0.89      0.94      0.92        18
     feature       0.96      0.96      0.96        28
 performance       0.89      0.94      0.92        18
    security       0.88      0.83      0.86        18

    accuracy                           0.93       120
   macro avg       0.92      0.92      0.92       120
weighted avg       0.93      0.93      0.92       120

Matriz de confusión (filas: real, columnas: predicción):
             bug  docs  feature  performance  security
bug           35     1        0            1         1
docs           0    17        1            0         0
feature        0     1       27            0         0
performance    0     0        0           17         1
security       2     0        0            1        15

Accuracy: 0.925 | F1 macro: 0.919 | F1 ponderado: 0.925

Modelo guardado en: …\models\triage_pipeline.joblib
Métricas guardadas en: …\docs\metrics.json
```

La ruta absoluta final cambia entre equipos. Las métricas se obtuvieron sobre la partición de prueba fija del dataset sintético actual.

### 3. Clasificar una incidencia con `--text`

```sh
uv run python src/predict.py --text "La API devuelve HTTP 500 al guardar un registro"
```

Carga `models/triage_pipeline.joblib`, limpia el texto con `clean_text()` y llama a `predict_proba()`. No modifica archivos. Muestra la categoría elegida, su probabilidad y el desglose de las cinco clases.

Salida real de la ejecución:

```text
Categoría predicha: bug
Confianza: 63.5%
Probabilidades por categoría:
  bug          63.5%
  security     12.0%
  docs         10.6%
  feature       7.0%
  performance   6.8%
```

Las probabilidades suman aproximadamente 100% (pueden existir diferencias mínimas de redondeo).

### 4. Ejecutar la demo

```sh
uv run python src/predict.py --demo
```

Carga el mismo modelo y clasifica cinco ejemplos incorporados en `src/predict.py`, uno por categoría objetivo. No escribe artefactos.

Salida real de la ejecución:

```text
Categoría esperada   Predicción       Confianza    Incidencia
-------------------------------------------------------------------------------------------------------------------
Bug                  bug                 40.1%     HTTP 500 y TypeError al guardar un registro en el endpoint de usuarios.
Feature              feature             52.4%     Solicito añadir filtros avanzados y exportación CSV para los reportes.
Performance          performance         71.0%     La consulta tarda varios segundos y presenta timeouts bajo carga alta.
Security             security            70.9%     Posible XSS por entrada sin validar y token de acceso visible en los logs.
Docs                 docs                75.8%     El README está desactualizado y falta un ejemplo de solicitud para la API.
```

### 5. Ejecutar las pruebas

```sh
uv run pytest -v
```

Ejecuta `tests/test_pipeline.py`, que verifica limpieza, entrenamiento e inferencia en memoria, entradas límite y el artefacto guardado. Lee `data/tickets_dataset.csv` y carga `models/triage_pipeline.joblib`; el resto de ajustes de prueba se ejecutan en memoria. Pytest puede crear o actualizar `.pytest_cache/`, que está excluido del control de versiones. La prueba del artefacto necesita que se haya ejecutado entrenamiento al menos una vez.

Resumen real:

```text
tests/test_pipeline.py::test_clean_text PASSED
tests/test_pipeline.py::test_pipeline_training_and_prediction PASSED
tests/test_pipeline.py::test_edge_cases_inference PASSED
tests/test_pipeline.py::test_saved_model_artifact PASSED
tests/test_pipeline.py::test_pipeline_has_expected_steps PASSED

============================== 5 passed in 2.05s ==============================
```

El tiempo de ejecución y la información de plataforma de pytest varían según el equipo.

## Solución de problemas frecuentes

### `uv` no se reconoce después de instalarlo

1. Cierra y vuelve a abrir PowerShell o la terminal para recargar `PATH`.
2. Comprueba `uv --version`.
3. Si sigue sin encontrarse, verifica la ubicación de instalación y que esa carpeta esté en `PATH`, según las instrucciones oficiales de Astral para tu sistema.

### Se ejecutó `predict.py` antes de entrenar

`predict.py` necesita `models/triage_pipeline.joblib`. Ejecuta:

```sh
uv run python src/train.py
```

Si tampoco existe el CSV, genera primero los datos:

```sh
uv run python src/generate_dataset.py
uv run python src/train.py
```

### Regenerar los datos y el modelo desde cero

Si cambias las plantillas, parámetros o lógica del generador, vuelve a ejecutar los pasos en orden:

```sh
uv sync
uv run python src/generate_dataset.py
uv run python src/train.py
uv run python src/predict.py --demo
uv run pytest -v
```

La generación reemplaza `data/tickets_dataset.csv`; el entrenamiento actualiza el modelo y las métricas. Revisa los cambios del CSV, del artefacto y de `docs/metrics.json` antes de incorporarlos al control de versiones.
