"""Pruebas de limpieza, entrenamiento e inferencia."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import (
    MODEL_PATH,
    explain_prediction,
    parse_args,
    predict_ticket,
    print_single_prediction,
)
from src.train import build_pipeline, top_positive_ngrams, train_and_evaluate
from src.utils import clean_text


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "tickets_dataset.csv")


def test_clean_text() -> None:
    assert clean_text("  HTTP 500   ERROR\t") == "http 500 error"
    assert clean_text("Consulta https://example.com/a?q=1 lenta") == "consulta lenta"
    assert clean_text(None) == ""
    assert clean_text("") == ""
    with pytest.raises(TypeError):
        clean_text(123)  # type: ignore[arg-type]


def test_pipeline_training_and_prediction(dataset: pd.DataFrame) -> None:
    small_dataset = dataset.groupby("category", group_keys=False).head(30).reset_index(drop=True)
    pipeline, _ = train_and_evaluate(small_dataset)

    label, confidence, probabilities = predict_ticket(
        pipeline, "HTTP 500 al guardar los datos"
    )

    assert label in set(small_dataset["category"])
    assert 0.0 <= confidence <= 1.0
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_explainability(dataset: pd.DataFrame, monkeypatch, capsys) -> None:
    pipeline, _ = train_and_evaluate(dataset)
    text = "HTTP 500 al guardar los datos"

    terms = explain_prediction(pipeline, text)
    assert 0 < len(terms) <= 3
    assert all(contribution > 0 for _, contribution in terms)
    assert all(term in text.lower() for term, _ in terms)
    assert all(
        len(ngrams) <= 5 for ngrams in top_positive_ngrams(pipeline).values()
    )

    print_single_prediction(pipeline, text, explain=True)
    assert "Términos con mayor contribución:" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["predict.py", "--demo", "--explain"])
    args = parse_args()
    assert args.demo and args.explain


def test_edge_cases_inference(dataset: pd.DataFrame) -> None:
    pipeline, _ = train_and_evaluate(dataset)
    for text in ("", "??!!@@", "quasar blorf zentangle"):
        label, confidence, probabilities = predict_ticket(pipeline, text)
        assert label in set(dataset["category"])
        assert np.isfinite(confidence)
        assert sum(probabilities.values()) == pytest.approx(1.0)


def test_saved_model_artifact() -> None:
    assert MODEL_PATH.exists()
    model = joblib.load(MODEL_PATH)
    examples = {
        "bug": "HTTP 500 y TypeError al guardar un registro.",
        "feature": "Añadir filtros avanzados y un nuevo endpoint de exportación.",
        "performance": "La consulta es lenta y presenta timeouts bajo carga.",
        "security": "Posible vulnerabilidad XSS y token expuesto en los logs.",
        "docs": "Actualizar el README y añadir un ejemplo de la API.",
    }
    for expected, text in examples.items():
        label, _, probabilities = predict_ticket(model, text)
        assert label == expected
        assert set(probabilities) == set(examples)


def test_pipeline_has_expected_steps() -> None:
    pipeline = build_pipeline()
    assert list(pipeline.named_steps) == ["tfidf", "classifier"]
