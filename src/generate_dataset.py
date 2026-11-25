"""Generación reproducible de un conjunto sintético de incidencias."""

import random
from pathlib import Path

import pandas as pd


RANDOM_SEED = 42
DATASET_SIZE = 600
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "tickets_dataset.csv"

TEMPLATES = {
    "bug": {
        "titles": [
            "Error al {action} en {component}",
            "Regresión en {component} tras el último despliegue",
            "La operación {action} devuelve un fallo inesperado",
            "Excepción intermitente en {component}",
            "HTTP 500 al {action} desde la aplicación",
            "La interfaz se rompe al {action}",
        ],
        "details": [
            "El usuario recibe una excepción {error} y la operación no termina.",
            "En producción aparece HTTP 500 al {action}; los logs muestran un stack trace.",
            "La pantalla queda en blanco después de {action}, aunque antes funcionaba.",
            "El comportamiento se reproduce en {environment} y bloquea el flujo habitual.",
            "La respuesta del servicio es incorrecta y provoca un fallo visible para el usuario.",
        ],
    },
    "feature": {
        "titles": [
            "Añadir {feature} para los usuarios",
            "Solicitud de mejora: {feature}",
            "Nuevo endpoint para {feature}",
            "Mejorar la experiencia de {component}",
            "Permitir {action} desde el panel",
            "Propuesta para ampliar {component}",
        ],
        "details": [
            "Sería útil permitir {action} sin salir de la pantalla actual.",
            "Se solicita una API nueva para que los clientes puedan {feature}.",
            "La interfaz debería ofrecer una opción para {feature} y confirmar el resultado.",
            "Esta mejora reduciría pasos manuales y facilitaría el flujo de trabajo.",
            "El comportamiento actual funciona, pero falta soporte para {feature}.",
        ],
    },
    "performance": {
        "titles": [
            "Consulta lenta al cargar {component}",
            "Alto consumo de memoria durante {action}",
            "Aumenta la latencia de {component}",
            "Timeout al procesar solicitudes grandes",
            "CPU elevada en el servicio de {component}",
            "Optimizar el rendimiento de {action}",
        ],
        "details": [
            "La consulta tarda varios segundos y el p95 supera el objetivo de latencia.",
            "El proceso consume demasiada memoria cuando aumenta el volumen de solicitudes.",
            "Se observan timeouts bajo carga; la respuesta se degrada al crecer el tráfico.",
            "La CPU permanece elevada y la cola de trabajos crece durante el procesamiento.",
            "El endpoint responde lentamente aunque no se registran excepciones en los logs.",
        ],
    },
    "security": {
        "titles": [
            "Posible vulnerabilidad {vulnerability} en {component}",
            "Token expuesto en los logs del servicio",
            "Permisos incorrectos al acceder a {component}",
            "Dependencia crítica pendiente de actualizar",
            "Revisar validación de entrada en {action}",
            "Acceso no autorizado a recursos de {component}",
        ],
        "details": [
            "Una entrada sin validar podría permitir {vulnerability} en el navegador.",
            "El token de acceso aparece en los logs y debe quedar redactado.",
            "Un usuario sin permisos puede consultar datos de otra cuenta.",
            "La dependencia afectada tiene una alerta crítica y requiere mitigación.",
            "La ruta no comprueba correctamente la autorización antes de devolver datos.",
        ],
    },
    "docs": {
        "titles": [
            "Actualizar la guía de {component}",
            "Falta un ejemplo para {feature} en la API",
            "Corregir instrucciones del README",
            "Documentación desactualizada para {action}",
            "Añadir pasos de configuración para {component}",
            "El ejemplo de {feature} no coincide con la API",
        ],
        "details": [
            "El README describe una opción que ya no existe y confunde a quienes empiezan.",
            "Falta un ejemplo completo de solicitud y respuesta para este endpoint.",
            "La guía de instalación necesita comandos actuales y una nota de configuración.",
            "El parámetro documentado no coincide con la respuesta actual del servicio.",
            "Se solicita aclarar los pasos de uso, con un ejemplo reproducible.",
        ],
    },
}

AMBIGUOUS_DETAILS = [
    "En una parte del flujo también se observa un timeout ocasional.",
    "El impacto depende del volumen de tráfico y del entorno de ejecución.",
    "Conviene revisar los logs y el comportamiento del endpoint durante el diagnóstico.",
    "El reporte apareció después de un cambio reciente en el servicio.",
]

VALUES = {
    "action": ["guardar un registro", "iniciar sesión", "cargar el historial", "exportar datos", "actualizar el perfil"],
    "component": ["panel de administración", "endpoint de usuarios", "servicio de pagos", "formulario de acceso", "API de reportes"],
    "error": ["TypeError", "ValueError", "KeyError", "RuntimeError"],
    "environment": ["Chrome", "Firefox", "el entorno de staging", "la última versión de la API"],
    "feature": ["filtros avanzados", "notificaciones", "exportación CSV", "paginación", "búsqueda por etiquetas"],
    "vulnerability": ["XSS", "SQLi", "inyección", "control de acceso roto"],
}

CATEGORY_COUNTS = {
    "bug": 180,
    "feature": 150,
    "performance": 90,
    "security": 90,
    "docs": 90,
}
AMBIGUOUS_LABEL_RATE = 0.10
RELATED_CATEGORIES = {
    "bug": ("performance", "security"),
    "feature": ("docs", "bug"),
    "performance": ("bug", "security"),
    "security": ("bug", "performance"),
    "docs": ("feature", "bug"),
}


def _render(template: str, rng: random.Random) -> str:
    values = {key: rng.choice(options) for key, options in VALUES.items()}
    return template.format(**values)


def generate_dataset(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Crea 600 tickets variados en un orden reproducible."""
    rng = random.Random(seed)
    rows = []
    ticket_id = 1

    for category, count in CATEGORY_COUNTS.items():
        templates = TEMPLATES[category]
        for _ in range(count):
            title = _render(rng.choice(templates["titles"]), rng)
            description = _render(rng.choice(templates["details"]), rng)
            if rng.random() < 0.18:
                description = f"{description} {rng.choice(AMBIGUOUS_DETAILS)}"
            rows.append(
                {
                    "id": f"TKT-{ticket_id:04d}",
                    "title": title,
                    "description": description,
                    "category": category,
                }
            )
            ticket_id += 1

    for row in rows:
        if rng.random() < AMBIGUOUS_LABEL_RATE:
            row["category"] = rng.choice(RELATED_CATEGORIES[row["category"]])

    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["id", "title", "description", "category"])


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset generado: {len(dataset)} incidencias en {OUTPUT_PATH}")
    print("Distribución por categoría:")
    print(dataset["category"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
