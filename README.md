# AI Issue Triage

Clasificador supervisado para apoyar el triaje de incidencias técnicas en equipos de desarrollo de software.

## Descripción del problema y objetivo del MVP

Los equipos reciben incidencias de distintos tipos y necesitan clasificarlas con rapidez para encaminarlas al área adecuada. El MVP aprende de ejemplos etiquetados para asignar cada nuevo ticket a `bug`, `feature`, `performance`, `security` o `docs`. Es una ayuda para priorizar y organizar el trabajo, no sustituye la revisión humana, especialmente ante incidencias de seguridad.

## Documentación técnica detallada

- [Arquitectura y funcionamiento interno](docs/ARQUITECTURA.md): flujo de datos, dataset, pipeline y lectura de métricas.
- [Guía de ejecución paso a paso](docs/GUIA_EJECUCION.md): instalación de `uv`, comandos, salidas verificadas y solución de problemas.

## Arquitectura y algoritmo

El MVP aplica aprendizaje supervisado mediante un `Pipeline` de scikit-learn con TF-IDF (n-gramas 1–2) y regresión logística. Una partición estratificada reserva el 20% de los 600 tickets para evaluación; el pipeline aprende el vocabulario solo con entrenamiento para evitar *data leakage*. El detalle de diseño se documenta en [Arquitectura](docs/ARQUITECTURA.md).

Métricas reales de la ejecución actual de `train.py`:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.925 |
| F1 macro | 0.919 |
| F1 ponderado | 0.925 |

El informe completo y la matriz de confusión se guardan en [`docs/metrics.json`](docs/metrics.json). Las métricas son de datos sintéticos y no predicen el rendimiento sobre incidencias reales.

## Evaluación crítica del enfoque de Machine Learning

TF-IDF con regresión logística ofrece entrenamiento e inferencia rápidos, bajo consumo de recursos y cierta interpretabilidad: los pesos positivos muestran qué n-gramas se asocian a cada categoría, y `--explain` destaca los términos que contribuyeron a una predicción. Sin embargo, el modelo se basa en coincidencias de palabras y no comprende bien el contexto. Por ejemplo, puede confundir solicitudes de documentación con un bug, como el ticket de prueba «documentación desactualizada para exportar datos…» (real: `bug`, predicha: `docs`). También confundió una incidencia de seguridad que mencionaba «consulta lenta» y «memoria» con `performance` (real: `security`); y una incidencia de `performance` que mencionaba un token expuesto fue predicha como `security`.

Las negaciones («no hay error») y el lenguaje ambiguo pueden invertir el sentido sin cambiar las palabras clave; el vocabulario fuera del dominio (*out-of-vocabulary*) no aporta características TF-IDF y deja poca evidencia para decidir. Además, estos errores proceden de una partición de prueba de datos sintéticos, así que las métricas y explicaciones requieren validación con tickets reales y revisión humana antes de usarse para priorizar, especialmente en seguridad.

## Investigación y mapeo crítico de herramientas de IA

| Herramienta y categoría | Integración en el flujo | Ventajas observables | Limitaciones y riesgos |
|---|---|---|---|
| GitHub Copilot Coding Agent / Codex — programación agéntica | Asignar una tarea acotada con criterios de aceptación; revisar cambios y pruebas en una rama o diff antes de integrar. | Puede acelerar tareas repetitivas, proponer implementaciones y trabajar con contexto del repositorio. | Puede alucinar APIs o criterios, introducir deuda técnica y requerir supervisión. El envío de código a servicios externos debe ajustarse a las políticas de privacidad y configuración de la organización. |
| Qodo (CodiumAI) — generación y análisis de pruebas | Usarlo para proponer casos límite y revisar cobertura; validar cada prueba contra el comportamiento esperado y los requisitos. | Ayuda a detectar escenarios omitidos y a generar esqueletos de pruebas con rapidez. | Puede crear pruebas acopladas a la implementación, mocks excesivos o asserts que solo confirman el código actual. Las pruebas generadas no sustituyen el diseño ni la revisión. |
| GitHub Projects / Issues + Copilot — gestión y trazabilidad | Registrar alcance, criterios de aceptación, decisiones y tareas; enlazar pull requests, cambios y revisiones con Issues/Projects. | Aporta visibilidad del progreso y trazabilidad desde el requisito hasta la implementación. | La calidad depende de mantener actualizados Issues y tableros; resúmenes automáticos pueden omitir contexto. Revisar privacidad, permisos y contenido compartido. |

Estas herramientas son complementarias: la IA propone y ayuda a ejecutar, mientras que las decisiones de diseño, la revisión de calidad y la responsabilidad de integrar siguen en manos del equipo.

### Herramientas utilizadas en este proyecto

Se utilizó **Gemini** para desarrollar la idea inicial, crear y refinar prompts, y resolver dudas durante la definición del proyecto. **Copilot Agents** se utilizó para realizar el proyecto: implementar sus componentes, documentación y pruebas, y ejecutar las verificaciones. Las propuestas generadas se contrastaron con la especificación y los resultados ejecutados.

## Distribución por fase del MVP

| Fase | Uso propuesto |
|---|---|
| Planificación | Gemini para concretar la idea, crear prompts y resolver dudas; Issues/Projects pueden ayudar a registrar alcance y seguimiento. |
| Desarrollo | Copilot Agents para realizar los componentes del proyecto; revisión de cambios y ejecución local de comandos. |
| Testing | Copilot Agents para apoyar la implementación de pruebas; pytest para verificar comportamiento y casos límite. Qodo es una alternativa evaluada, no una herramienta utilizada en este proyecto. |
| Documentación | Copilot Agents para apoyar la redacción del README; revisión de que comandos, salidas y métricas coincidan con el proyecto. |

## Requisitos, instalación y ejecución

Se requiere Python 3.11 o superior y [`uv`](https://docs.astral.sh/uv/). Desde la raíz del proyecto, sincroniza el entorno y ejecuta los pasos principales:

```bash
uv sync
uv run python src/generate_dataset.py
uv run python src/train.py
uv run python src/predict.py --demo
uv run pytest -v
```

Para clasificar un texto individual:

```bash
uv run python src/predict.py --text "La API devuelve HTTP 500 al guardar un registro"
```

El entrenamiento genera `models/triage_pipeline.joblib` y actualiza `docs/metrics.json`; `uv.lock` fija las dependencias. Consulta la [Guía de ejecución](docs/GUIA_EJECUCION.md) para instalación, explicación de los comandos, ejemplos completos de salidas y solución de problemas.

## Bitácora de desarrollo asistido por IA

| Tarea delegada a la IA | Riesgo metodológico detectado | Decisión y corrección aplicada |
|---|---|---|
| Definir y refinar la idea, crear prompts y resolver dudas (Gemini). | Las propuestas iniciales pueden ser ambiguas o no ajustarse al alcance académico. | Usar Gemini como apoyo de ideación y aclaración; contrastar y concretar los requisitos en la especificación. |
| Realizar el proyecto: generación de datos, pipeline de clasificación, CLI, pruebas y documentación (Copilot Agents). | La vectorización antes del split provoca *data leakage*; plantillas sintéticas triviales podrían producir accuracy perfecto; mocks excesivos pueden acoplar tests a la implementación. | Encapsular TF-IDF y el clasificador en `Pipeline`, introducir ambigüedad controlada en los datos y verificar el comportamiento con pytest e inferencia real. |
| Revisar resultados y documentación generados con asistencia de IA. | Las salidas pueden contener errores o afirmar resultados no verificados. | Ejecutar los comandos documentados y reflejar en el README las métricas obtenidas, con sus limitaciones explícitas. |

Gemini se empleó para la ideación, la creación de prompts y la resolución de dudas; Copilot Agents, para la realización del proyecto. En ambos casos, la asistencia se trató como apoyo y las salidas se verificaron contra los requisitos y las pruebas antes de incorporarlas.

## Verificación y evidencias (capturas)

Ejecuta los comandos de la sección anterior y guarda capturas legibles de la terminal en `docs/screenshots/` con estos nombres:

![Entrenamiento y métricas](docs/screenshots/train_metrics.png)

![Demo CLI](docs/screenshots/predict_demo.png)

![Pytest](docs/screenshots/pytest_pass.png)
