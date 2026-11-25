"""Funciones compartidas de procesamiento de texto."""

import re


_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_NON_WORD_PATTERN = re.compile(r"[^\w\s#./-]", flags=re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Normaliza texto técnico conservando códigos y tokens relevantes."""
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError("El texto de la incidencia debe ser una cadena.")

    text = _URL_PATTERN.sub(" ", text.lower())
    text = _NON_WORD_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()
