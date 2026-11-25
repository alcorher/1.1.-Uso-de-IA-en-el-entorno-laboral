"""Interfaz de línea de comandos para clasificar incidencias."""

import argparse
from pathlib import Path

import joblib

try:
    from .utils import clean_text
except ImportError:
    from utils import clean_text


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "triage_pipeline.joblib"
DEMO_TICKETS = [
    ("Bug", "HTTP 500 y TypeError al guardar un registro en el endpoint de usuarios."),
    ("Feature", "Solicito añadir filtros avanzados y exportación CSV para los reportes."),
    ("Performance", "La consulta tarda varios segundos y presenta timeouts bajo carga alta."),
    ("Security", "Posible XSS por entrada sin validar y token de acceso visible en los logs."),
    ("Docs", "El README está desactualizado y falta un ejemplo de solicitud para la API."),
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo {MODEL_PATH}. "
            "Entrénalo con `uv run python src/train.py`."
        )
    return joblib.load(MODEL_PATH)


def predict_ticket(model, text: str) -> tuple[str, float, dict[str, float]]:
    """Devuelve etiqueta, confianza y probabilidades para un ticket."""
    cleaned = clean_text(text)
    probabilities = model.predict_proba([cleaned])[0]
    classes = model.named_steps["classifier"].classes_
    probability_by_class = {
        label: float(probability)
        for label, probability in zip(classes, probabilities)
    }
    label = max(probability_by_class, key=probability_by_class.get)
    return label, probability_by_class[label], probability_by_class


def explain_prediction(model, text: str, limit: int = 3) -> list[tuple[str, float]]:
    """Devuelve los términos presentes con mayor contribución positiva."""
    label, _, _ = predict_ticket(model, text)
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    class_index = list(classifier.classes_).index(label)
    vector = vectorizer.transform([clean_text(text)])
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[class_index]

    contributions = [
        (str(feature_names[index]), float(value * coefficients[index]))
        for index, value in zip(vector.indices, vector.data)
        if value * coefficients[index] > 0
    ]
    return sorted(contributions, key=lambda item: item[1], reverse=True)[:limit]


def print_explanation(model, text: str) -> None:
    terms = explain_prediction(model, text)
    if terms:
        print("Términos con mayor contribución:")
        for term, contribution in terms:
            print(f"  {term}: {contribution:.3f}")
    else:
        print("No hay términos con contribución positiva para explicar esta predicción.")


def print_single_prediction(model, text: str, explain: bool = False) -> None:
    label, confidence, probabilities = predict_ticket(model, text)
    print(f"Categoría predicha: {label}")
    print(f"Confianza: {confidence:.1%}")
    print("Probabilidades por categoría:")
    for category, probability in sorted(
        probabilities.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"  {category:<12} {probability:.1%}")
    if explain:
        print_explanation(model, text)


def print_demo(model, explain: bool = False) -> None:
    print(f"{'Categoría esperada':<20} {'Predicción':<16} {'Confianza':<12} Incidencia")
    print("-" * 115)
    for expected, text in DEMO_TICKETS:
        label, confidence, _ = predict_ticket(model, text)
        print(f"{expected:<20} {label:<16} {confidence:>8.1%}     {text}")
        if explain:
            print_explanation(model, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clasificador de incidencias técnicas.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", help="Texto de una incidencia para clasificar.")
    mode.add_argument("--demo", action="store_true", help="Ejecuta cinco ejemplos.")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Muestra los términos que más contribuyen a la categoría predicha.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        model = load_model()
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    if args.demo:
        print_demo(model, explain=args.explain)
    else:
        print_single_prediction(model, args.text, explain=args.explain)


if __name__ == "__main__":
    main()
