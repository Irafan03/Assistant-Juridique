"""Extraction de texte depuis PDF et DOCX, avec OCR pour les pages scannées.

Les métadonnées extraites automatiquement sont marquées ``auto``; seules
les valeurs confirmées par une révision humaine peuvent être ajoutées à
``verified_metadata``.  Le module ne fabrique aucune métadonnée absente
du document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DOCUMENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "circulaire": ["circulaire", "منشور", "lettre circulaire"],
    "loi":        ["loi ", "loi\n", "قانون", "code ", "code\n", "dahir portant loi",
                   "dahir portant promulgation de la loi"],
    "décret":     ["décret", "decret", "مرسوم"],
    "arrêté":     ["arrêté", "arrete", "قرار"],
    "directive":  ["directive"],
    "dahir":      ["dahir n°", "dahir n°", "dahir portant", "dahir loi"],
    "article":    [],          # jamais le type de document racine
    "texte":      [],          # fallback explicite
}


@dataclass
class RawPage:
    """Texte brut d'une page avec sa provenance."""

    page_number: int       # 1-indexed
    text: str
    from_ocr: bool = False
    width_pt: float = 0.0
    height_pt: float = 0.0


@dataclass
class RawDocument:
    """Document brut extrait d'un fichier PDF ou DOCX."""

    source_file: str          # chemin absolu
    document_id: str          # dérivé du nom de fichier, stable
    pages: list[RawPage] = field(default_factory=list)
    full_text: str = ""
    detected_type: str | None = None      # auto-détecté
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def auto_metadata_keys(self) -> tuple[str, ...]:
        """Clés dont la valeur a été extraite automatiquement (non vérifiées)."""
        return tuple(self.metadata.get("_auto_keys", []))


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _document_id_from_path(path: Path) -> str:
    """Identifiant stable dérivé du nom de fichier, sans extension."""
    return re.sub(r"[^\w\-]", "_", path.stem.lower())


def _detect_type(text: str) -> str | None:
    """Détection heuristique du type de document sur les premiers 2000 caractères."""
    sample = text[:2000].lower()
    for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items():
        if doc_type in ("article", "texte"):
            continue
        if any(kw in sample for kw in keywords):
            return doc_type
    return None


def _extract_title(text: str) -> str | None:
    """Tente d'extraire le titre depuis les premières lignes non vides."""
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) > 10 and not stripped.startswith(("#", "-", "•")):
            return stripped[:250]
    return None


def _extract_date(text: str) -> str | None:
    """Extrait la première date au format JJ/MM/AAAA ou AAAA-MM-JJ rencontrée."""
    patterns = [
        re.compile(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})\b"),
        re.compile(r"\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b"),
        re.compile(r"\b(\d{1,2}\s+\w+\s+\d{4})\b"),          # "12 janvier 2021"
    ]
    for pat in patterns:
        match = pat.search(text[:3000])
        if match:
            return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Extraction PDF
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path, *, ocr_lang: str = "fra+ara", min_text_len: int = 50) -> RawDocument:
    """Extrait le texte d'un PDF.  Recourt à l'OCR si une page est presque vide."""
    try:
        # PyMuPDF >= 2.0 s'importe sous 'pymupdf' ; les versions antérieures sous 'fitz'
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # type: ignore[no-redef]
    except ImportError as exc:
        raise ImportError("pymupdf est requis : pip install pymupdf") from exc

    doc = fitz.open(str(path))
    doc_id = _document_id_from_path(path)
    pages: list[RawPage] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")
        from_ocr = False

        if len(text.strip()) < min_text_len:
            # Page probablement scannée — OCR
            try:
                import pytesseract
                from PIL import Image
                import io

                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, lang=ocr_lang)
                from_ocr = True
            except Exception:
                # OCR non disponible : on conserve le texte vide
                pass

        rect = page.rect
        pages.append(RawPage(
            page_number=page_index + 1,
            text=text,
            from_ocr=from_ocr,
            width_pt=rect.width,
            height_pt=rect.height,
        ))

    full_text = "\n".join(p.text for p in pages)
    detected_type = _detect_type(full_text)
    title = _extract_title(full_text)
    date = _extract_date(full_text)

    return RawDocument(
        source_file=str(path.resolve()),
        document_id=doc_id,
        pages=pages,
        full_text=full_text,
        detected_type=detected_type,
        metadata={
            "title": title,
            "date": date,
            "num_pages": len(pages),
            "format": "pdf",
            "_auto_keys": ["title", "date", "detected_type"],
        },
    )


# ---------------------------------------------------------------------------
# Extraction DOCX
# ---------------------------------------------------------------------------

def _extract_docx(path: Path) -> RawDocument:
    """Extrait le texte d'un fichier DOCX paragraphe par paragraphe."""
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise ImportError("python-docx est requis : pip install python-docx") from exc

    docx = DocxDocument(str(path))
    lines: list[str] = []
    for para in docx.paragraphs:
        text = para.text.strip()
        if text:
            lines.append(text)

    full_text = "\n".join(lines)
    doc_id = _document_id_from_path(path)
    detected_type = _detect_type(full_text)
    title = _extract_title(full_text)
    date = _extract_date(full_text)

    # DOCX n'a pas de pagination native ; on découpe arbitrairement à 3000 chars
    chunk_size = 3000
    pages: list[RawPage] = []
    for i, start in enumerate(range(0, max(len(full_text), 1), chunk_size), start=1):
        pages.append(RawPage(page_number=i, text=full_text[start:start + chunk_size]))

    return RawDocument(
        source_file=str(path.resolve()),
        document_id=doc_id,
        pages=pages,
        full_text=full_text,
        detected_type=detected_type,
        metadata={
            "title": title,
            "date": date,
            "num_pages": len(pages),
            "format": "docx",
            "_auto_keys": ["title", "date", "detected_type"],
        },
    )


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def extract_document(path: str | Path, *, ocr_lang: str = "fra+ara") -> RawDocument:
    """Extrait un document PDF ou DOCX.

    Args:
        path: Chemin vers le fichier.
        ocr_lang: Langues Tesseract séparées par ``+``.  ``fra+ara`` par défaut.

    Returns:
        :class:`RawDocument` avec ``detected_type`` et ``metadata`` automatiques.
        Les clés ``metadata['_auto_keys']`` listent les champs non vérifiés.

    Raises:
        ValueError: Si le format n'est pas supporté.
        FileNotFoundError: Si le fichier n'existe pas.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path, ocr_lang=ocr_lang)
    if suffix in (".docx", ".doc"):
        return _extract_docx(path)
    raise ValueError(f"Format non supporté : {suffix!r}. Seuls PDF et DOCX sont acceptés.")
