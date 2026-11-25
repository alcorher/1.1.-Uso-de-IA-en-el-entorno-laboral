# Arquitectura y funcionamiento interno

## Visión general y flujo de datos

El proyecto genera incidencias sintéticas etiquetadas, entrena un clasificador supervisado y serializa todo el preprocesamiento aprendido junto al modelo. La inferencia carga ese artefacto y aplica la misma transformación TF-IDF a entradas nuevas.

```text
src/generate_dataset.py
        │
        └── genera 600 filas reproducibles
                    ▼
        data/tickets_dataset.csv
                    │
                    ▼
             src/train.py
        combina título + descripción
        limpia con src/utils.py
        divide train/test (80/20)
                    │
                    ▼
       sklearn Pipeline (fit solo en train)
       ├── TfidfVectorizer
       └── LogisticRegression
             │                 │
             ▼                 ▼
models/triage_pipeline.joblib  docs/metrics.json
             │
             ▼
        src/predict.py
        limpia el texto de entrada
        predice etiqueta y probabilidades
```

`predict.py` requiere que el modelo exista. Por tanto, en una instalación nueva se debe ejecutar `train.py` antes de la predicción.

## Diseño y estructura del dataset

`data/tickets_dataset.csv` contiene las columnas siguientes:

| Columna | Contenido |
|---|---|
| `id` | Identificador reproducible con formato `TKT-0001`. |
| `title` | Título técnico generado a partir de plantillas. |
| `description` | Descripción técnica generada a partir de plantillas y frases de contexto. |
| `category` | Etiqueta supervisada: `bug`, `feature`, `performance`, `security` o `docs`. |

El generador declara cuotas iniciales de 180 `bug`, 150 `feature` y 90 por cada una de las otras tres categorías. Luego reasigna determinísticamente el 10% de las etiquetas a categorías relacionadas. Además, el 18% de las descripciones recibe una frase ambigua (por ejemplo, un timeout o una referencia a logs). Así se evita que todas las palabras identifiquen de forma trivial una sola clase.

Con la semilla `42`, la distribución final del CSV es:

| Categoría | Tickets | Porcentaje |
|---|---:|---:|
| `bug` | 188 | 31,3 % |
| `feature` | 140 | 23,3 % |
| `performance` | 91 | 15,2 % |
| `security` | 91 | 15,2 % |
| `docs` | 90 | 15,0 % |
| **Total** | **600** | **100 %** |

Las cifras resultan del proceso de reasignación de etiquetas; por eso pueden diferir de las cuotas previas a dicha operación. La generación es determinista para una misma versión del código y semilla.

## Desglose del pipeline de aprendizaje automático

### Preprocesamiento: `src/utils.py`

`clean_text(text: str | None) -> str` procesa el texto en este orden:

1. Devuelve `""` para `None`; una cadena vacía también termina como `""`.
2. Lanza `TypeError` si la entrada no es una cadena ni `None`.
3. Convierte el texto a minúsculas.
4. Elimina las URLs que empiezan por `http://`, `https://` o `www.` hasta el siguiente espacio.
5. Sustituye por espacios los caracteres que no sean letras/números Unicode, espacios o los símbolos conservados `#`, `.`, `/` y `-`.
6. Colapsa espacios repetidos y elimina espacios al principio y al final.

Ejemplo:

```text
Entrada: "  HTTP 500: ERROR al guardar https://api.example.com/v1?id=3!!! "
Salida:  "http 500 error al guardar"
```

Se conservan códigos numéricos, nombres de excepciones, tokens como `xss` y `sqli`, y separadores técnicos permitidos. En entrenamiento, `prepare_texts()` concatena `title`, un espacio y `description`; reemplaza valores nulos de esas columnas por cadenas vacías y aplica `clean_text`.

### Vectorización: `TfidfVectorizer`

`build_pipeline()` configura el vectorizador con:

| Parámetro | Valor | Efecto |
|---|---|---|
| `ngram_range` | `(1, 2)` | Considera términos individuales y pares consecutivos de términos. |
| `max_features` | `3000` | Limita el vocabulario aprendido a un máximo de 3000 características. |
| `sublinear_tf` | `True` | Usa una escala logarítmica de frecuencia de término para reducir el efecto de repeticiones. |

Al ajustar el pipeline con los textos de entrenamiento, TF-IDF aprende el vocabulario y los pesos de los términos. Cada incidencia se transforma en un vector numérico disperso; la mayoría de sus posiciones son cero porque cada texto solo contiene una pequeña parte del vocabulario.

### Clasificador: `LogisticRegression`

La regresión logística multinomial está configurada con `max_iter=1000`, `class_weight="balanced"` y `random_state=42`. El ajuste de pesos compensa diferencias en el número de ejemplos por clase; `random_state` hace reproducibles los componentes que dependan de aleatoriedad.

Tras el ajuste, `predict_proba()` devuelve una probabilidad para cada categoría. `src/predict.py` asocia esas probabilidades con `model.named_steps["classifier"].classes_`; la clase con el valor mayor es la predicción y su valor se presenta como confianza. Son probabilidades del modelo, no una garantía de que la predicción sea correcta.

### Prevención de *data leakage*

`train_and_evaluate()` divide los textos y etiquetas con `train_test_split(test_size=0.2, random_state=42, stratify=labels)`. Solo después ajusta el `Pipeline` con `x_train` y `y_train`. El vectorizador está dentro del pipeline, así que su vocabulario y los pesos IDF se aprenden al llamar a `pipeline.fit(x_train, y_train)`, nunca con `x_test`. La prueba queda reservada para una evaluación que no ha influido en el ajuste del vocabulario o del clasificador.

La limpieza textual es determinista y no estima estadísticas sobre el conjunto completo; aplicar esa normalización antes del split no introduce el tipo de fuga que se evita con el vectorizador.

## Interpretación de métricas y artefactos

La última ejecución de `uv run python src/train.py`, sobre 600 tickets y una partición de prueba de 120, produjo:

| Categoría | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `bug` | 0.95 | 0.92 | 0.93 | 38 |
| `docs` | 0.89 | 0.94 | 0.92 | 18 |
| `feature` | 0.96 | 0.96 | 0.96 | 28 |
| `performance` | 0.89 | 0.94 | 0.92 | 18 |
| `security` | 0.88 | 0.83 | 0.86 | 18 |
| **macro avg** | **0.92** | **0.92** | **0.92** | **120** |
| **weighted avg** | **0.93** | **0.93** | **0.92** | **120** |

Accuracy: **0.925** · F1 macro: **0.919** · F1 ponderado: **0.925**.

- **Precision** indica, para una clase, qué proporción de las predicciones de esa clase eran correctas.
- **Recall** indica qué proporción de los ejemplos reales de esa clase se recuperó.
- **F1-score** es la media armónica de precision y recall.
- **Macro avg** promedia las métricas de las clases dando el mismo peso a cada una; **weighted avg** las pondera por su soporte (`support`).

La matriz de confusión muestra las filas como clase real y las columnas como predicción:

| Real \ Predicción | bug | docs | feature | performance | security |
|---|---:|---:|---:|---:|---:|
| **bug** | 35 | 1 | 0 | 1 | 1 |
| **docs** | 0 | 17 | 1 | 0 | 0 |
| **feature** | 0 | 1 | 27 | 0 | 0 |
| **performance** | 0 | 0 | 0 | 17 | 1 |
| **security** | 2 | 0 | 0 | 1 | 15 |

Los artefactos generados son:

- `models/triage_pipeline.joblib`: pipeline ajustado, incluyendo vocabulario TF-IDF y clasificador.
- `docs/metrics.json`: métricas como JSON, incluyendo `accuracy`, `f1_macro`, `f1_weighted`, `classification_report`, `confusion_matrix` y `test_size`.
- `data/tickets_dataset.csv`: dataset que consume el entrenamiento y regenera `generate_dataset.py`.

Las métricas describen únicamente este dataset sintético y esta partición fija; no equivalen a una evaluación con incidencias reales.
