"""Conservative extraction of explicit legal reference strings."""

from __future__ import annotations

import re

# Capture reference-shaped strings only; resolution is performed by exact normalized
# matching against references actually present in indexed metadata/text.
REFERENCE_PATTERNS = (
    re.compile(r"\b(?:loi|décret|decret|arrêté|arrete|circulaire)\s*(?:n[°ºo.]\s*)?\d[\w./-]*", re.I),
    re.compile(r"\b(?:articles?|art\.)\s*\d+(?:\s*(?:bis|ter|quater))?(?:\s*(?:à|a|et|-)\s*\d+)?", re.I),
    re.compile(r"(?:قانون|مرسوم|قرار|منشور)\s*(?:رقم\s*)?[\d٠-٩]+[\w\d٠-٩./-]*", re.I),
    re.compile(r"(?:المادة|مادة)\s*[\d٠-٩]+(?:\s*(?:إلى|الى|و|-)\s*[\d٠-٩]+)?", re.I),
)


def normalize_reference(value: str) -> str:
    value = value.casefold().replace("º", "°")
    value = re.sub(r"\barticles\b", "article", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*([°./-])\s*", r"\1", value)
    return value.strip(" .,:;()[]")


def extract_references(text: str) -> list[str]:
    found: dict[str, str] = {}
    for pattern in REFERENCE_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(0).strip()
            found.setdefault(normalize_reference(raw), raw)
    return list(found.values())
