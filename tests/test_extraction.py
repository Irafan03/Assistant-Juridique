"""Tests du module d'extraction (sans PDF réel ni OCR)."""

from __future__ import annotations

import textwrap
from pathlib import Path
import tempfile
import os

from legal_rag.extraction import (
    _detect_type,
    _extract_date,
    _extract_title,
    _document_id_from_path,
    extract_document,
)


def test_detect_type_circulaire():
    assert _detect_type("Circulaire n° 5 du Gouverneur") == "circulaire"


def test_detect_type_decret():
    assert _detect_type("Décret n° 2-20-331 portant sur…") == "décret"


def test_detect_type_loi():
    assert _detect_type("LOI n° 12-03 relative à…") == "loi"


def test_detect_type_arrete():
    assert _detect_type("Arrêté du ministre des finances") == "arrêté"


def test_detect_type_arabic_circular():
    assert _detect_type("منشور رقم 5 بشأن…") == "circulaire"


def test_detect_type_unknown_returns_none():
    assert _detect_type("Rapport annuel de performance") is None


def test_extract_date_slash():
    assert _extract_date("En date du 12/03/2021 il a été décidé") == "12/03/2021"


def test_extract_date_dash():
    assert _extract_date("Référence 2019-06-15 relative à") == "2019-06-15"


def test_extract_date_none():
    assert _extract_date("Aucune date dans ce texte") is None


def test_extract_title_basic():
    text = "Circulaire n° 5\nObjet : congé annuel"
    title = _extract_title(text)
    assert title is not None and "Circulaire" in title


def test_document_id_from_path():
    doc_id = _document_id_from_path(Path("Recueil textes 2021.pdf"))
    # Doit être lowercase alphanumérique avec underscores uniquement
    assert doc_id == "recueil_textes_2021"
    assert " " not in doc_id


def test_extract_document_unsupported_format(tmp_path):
    bad_file = tmp_path / "document.txt"
    bad_file.write_text("contenu")
    try:
        extract_document(bad_file)
        assert False, "ValueError attendu"
    except ValueError as exc:
        assert ".txt" in str(exc)


def test_extract_document_missing_file():
    try:
        extract_document("/chemin/inexistant/doc.pdf")
        assert False, "FileNotFoundError attendu"
    except FileNotFoundError:
        pass
