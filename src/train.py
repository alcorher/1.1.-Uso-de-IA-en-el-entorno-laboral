"""Entrenamiento, evaluación y serialización del clasificador."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from .utils import clean_text
except ImportError:
    from utils import clean_text


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tickets_dataset.csv"
MODEL_PATH = ROOT / "models" / "triage_pipeline.joblib"
METRICS_PATH = ROOT / "docs" / "metrics.json"


def build_pipeline() -> Pipeline:
    """Construye el pipeline completo para evitar fuga de datos."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=3000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def prepare_texts(dataset: pd.DataFrame) -> pd.Series:
    """Combina título y descripción y normaliza cada incidencia."""
    return (dataset["title"].fillna("") + " " + dataset["description"].fillna("")).map(clean_text)


def top_positive_ngrams(
    pipeline: Pipeline, limit: int = 5
) -> dict[str, list[tuple[str, float]]]:
    """Devuelve los n-gramas con mayor peso positivo de cada categoría."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    result = {}

    for label, coefficients in zip(classifier.classes_, classifier.coef_):
        positive_indices = [
            index for index, coefficient in enumerate(coefficients) if coefficient > 0
        ]
        top_indices = sorted(
            positive_indices, key=lambda index: coefficients[index], reverse=True
        )[:limit]
        result[label] = [
            (str(feature_names[index]), float(coefficients[index]))
            for index in top_indices
        ]

    return result


def print_top_positive_ngrams(pipeline: Pipeline) -> None:
    """Imprime los n-gramas más asociados a cada categoría."""
    print("\nN-gramas con mayor peso positivo por categoría:")
    for label, ngrams in top_positive_ngrams(pipeline).items():
        terms = ", ".join(f"{term} ({weight:.3f})" for term, weight in ngrams)
        print(f"  {label:<12} {terms}")


def train_and_evaluate(dataset: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Entrena el pipeline y devuelve métricas en datos de prueba reservados."""
    texts = prepare_texts(dataset)
    labels = dataset["category"]
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    class_labels = sorted(labels.unique())
    report = classification_report(
        y_test,
        predictions,
        labels=class_labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=class_labels)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": {
            "labels": class_labels,
            "values": matrix.tolist(),
        },
        "test_size": len(y_test),
    }

    print("Reporte de clasificación:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=class_labels,
            zero_division=0,
        )
    )
    print("Matriz de confusión (filas: real, columnas: predicción):")
    print(pd.DataFrame(matrix, index=class_labels, columns=class_labels).to_string())
    print(
        f"\nAccuracy: {metrics['accuracy']:.3f} | "
        f"F1 macro: {metrics['f1_macro']:.3f} | "
        f"F1 ponderado: {metrics['f1_weighted']:.3f}"
    )
    print_top_positive_ngrams(pipeline)
    print("\nEjemplos mal clasificados (máximo 3):")
    shown = 0
    for text, actual, predicted in zip(x_test, y_test, predictions):
        if actual == predicted:
            continue
        summary = text if len(text) <= 120 else f"{text[:117]}..."
        print(f"  Texto: {summary}")
        print(f"  Real: {actual} | Predicha: {predicted}")
        shown += 1
        if shown == 3:
            break
    if shown == 0:
        print("  Ninguno.")
    return pipeline, metrics


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Ejecuta primero "
            "`uv run python src/generate_dataset.py`."
        )

    dataset = pd.read_csv(DATA_PATH)
    pipeline, metrics = train_and_evaluate(dataset)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nModelo guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    main()
