"""Trois stratégies de chunking pour les documents juridiques.

Chaque stratégie retourne une liste de :class:`~legal_rag.models.Chunk` prêts
à être indexés.  Les relations parent–enfant sont conservées dans ``parent_id``.
Les métadonnées extraites automatiquement sont présentes dans ``metadata``
mais non listées dans ``verified_metadata``.

Stratégies :
- ``structural``   : découpage par articles/sections détectés dans le texte.
- ``hierarchical`` : arbre document → section → sous-section → paragraphe.
- ``parent_child`` : segments larges (parent) + sous-segments plus fins (enfants).
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from .extraction import RawDocument, RawPage
from .models import Chunk
from .references import extract_references

# ---------------------------------------------------------------------------
# Patterns d'identification d'articles / sections
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(
    r"(?:^|\n)"
    r"(?:"
    # Cas 1 — article propre : 1 à 3 chiffres (± bis/ter/quater) suivi d'un séparateur.
    r"(?:article|art\.)\s*(\d{1,3}(?:\s*(?:bis|ter|quater))?)"
    r"(?=\s|$|\n)"
    # Cas 2 — article corrompu par une note de bas de page collée :
    # « Article 14235 » = Article 14 + note 235. On capture les 1-2 premiers
    # chiffres (le vrai numéro d'article) et on ignore les 3 chiffres suivants
    # (le numéro de note), exigeant un séparateur après. Les vrais articles
    # AML de la circulaire 5/W/2017 sont ≤ 99 (2 chiffres) ; les notes du
    # recueil sont à 3 chiffres (235, 238, 239, 241…).
    r"|(?:article|art\.)\s*(\d{1,2})\d{3}(?=\s|\n|[A-Za-zÀ-ÿ])"
    # Cas 3 — article arabe : المادة ٤ (1-3 chiffres)
    r"|المادة\s*([\d٠-٩]{1,3})"
    r")",
    re.I | re.MULTILINE,
)


def _article_number_from_match(m: re.Match) -> str | None:
    """Extrait le numéro d'article propre depuis un match de _ARTICLE_RE.

    Groupe 1 : article propre (cas 1). Groupe 2 : article corrompu (cas 2).
    Groupe 3 : article arabe (cas 3).
    """
    if m.group(1):
        return m.group(1).strip()
    if m.group(2):
        return m.group(2).strip()
    if m.group(3):
        return m.group(3).strip()
    return None

_SECTION_RE = re.compile(
    r"(?:^|\n)"
    r"(?:chapitre|section|titre|partie)\s*[\dIVXivx]+",
    re.I | re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Utilitaires communs
# ---------------------------------------------------------------------------

def _make_id(*parts: str) -> str:
    """Identifiant déterministe : concatène les parties avec ``::``."""
    base = "::".join(parts)
    # On utilise uuid5 (SHA-1) pour un identifiant stable et court.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def _page_of(offset: int, pages: list[RawPage]) -> int | None:
    """Numéro de page (1-indexed) pour un offset dans le texte complet."""
    pos = 0
    for page in pages:
        pos += len(page.text)
        if offset < pos:
            return page.page_number
    return pages[-1].page_number if pages else None


def _base_meta(doc: RawDocument) -> dict[str, Any]:
    return {
        "date": doc.metadata.get("date"),
        "format": doc.metadata.get("format"),
        "from_ocr": any(p.from_ocr for p in doc.pages),
        "source": "auto",
    }


def _chunk_from(
    doc: RawDocument,
    chunk_id: str,
    text: str,
    *,
    parent_id: str | None = None,
    article_number: str | None = None,
    page_number: int | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Chunk:
    refs = tuple(extract_references(text))
    meta = {**_base_meta(doc), **(extra_meta or {})}
    return Chunk(
        document_id=doc.document_id,
        chunk_id=chunk_id,
        text=text.strip(),
        document_type=doc.detected_type,
        title=doc.metadata.get("title"),
        parent_id=parent_id,
        source_file=doc.source_file,
        page_number=page_number,
        article_number=article_number,
        references=refs,
        metadata=meta,
        # Aucun champ n'est marqué vérifié ici : c'est le rôle de l'annotateur.
        verified_metadata=(),
    )


# ---------------------------------------------------------------------------
# Stratégie 1 : Chunking structurel
# ---------------------------------------------------------------------------

def chunk_structural(doc: RawDocument, *, max_chars: int = 1500) -> list[Chunk]:
    """Découpe par articles détectés, puis par taille maximale.

    - Un chunk par article identifié dans le texte.
    - Si un article dépasse ``max_chars``, il est subdivisé en paragraphes.
    - Les chunks orphelins (avant le premier article) constituent le chunk
      d'en-tête du document.
    """
    text = doc.full_text
    splits: list[tuple[str, str | None, int]] = []   # (texte, numéro_article, offset)

    boundaries: list[tuple[int, str | None]] = [(0, None)]
    for m in _ARTICLE_RE.finditer(text):
        art_num = _article_number_from_match(m)
        boundaries.append((m.start(), art_num or None))

    for i, (start, art_num) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        segment = text[start:end].strip()
        if segment:
            splits.append((segment, art_num, start))

    chunks: list[Chunk] = []
    for seg_text, art_num, offset in splits:
        page = _page_of(offset, doc.pages)
        if len(seg_text) <= max_chars:
            cid = _make_id(doc.document_id, "structural", art_num or str(offset))
            chunks.append(_chunk_from(doc, cid, seg_text, article_number=art_num, page_number=page))
        else:
            # Subdivision en paragraphes
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", seg_text) if p.strip()]
            buf, buf_offset = "", offset
            for para in paragraphs:
                if buf and len(buf) + len(para) > max_chars:
                    cid = _make_id(doc.document_id, "structural", art_num or str(buf_offset), str(len(chunks)))
                    chunks.append(_chunk_from(doc, cid, buf, article_number=art_num, page_number=_page_of(buf_offset, doc.pages)))
                    buf = para
                    buf_offset = offset + seg_text.find(para)
                else:
                    buf = (buf + "\n" + para).strip() if buf else para
            if buf:
                cid = _make_id(doc.document_id, "structural", art_num or str(buf_offset), "last")
                chunks.append(_chunk_from(doc, cid, buf, article_number=art_num, page_number=_page_of(buf_offset, doc.pages)))

    return chunks


# ---------------------------------------------------------------------------
# Stratégie 2 : RAG hiérarchique
# ---------------------------------------------------------------------------

def chunk_hierarchical(doc: RawDocument, *, max_leaf_chars: int = 800) -> list[Chunk]:
    """Arbre document → section → sous-section (article) → feuille (paragraphe).

    Niveaux :
    - ``doc``     : chunk racine représentant tout le document.
    - ``section`` : chapitres / titres / parties.
    - ``article`` : articles détectés.
    - ``leaf``    : paragraphes à l'intérieur d'un article.
    """
    text = doc.full_text
    chunks: list[Chunk] = []

    # Chunk racine (document entier)
    root_id = _make_id(doc.document_id, "hierarchical", "root")
    root = _chunk_from(doc, root_id, text[:2000], page_number=1,
                       extra_meta={"level": "doc", "is_root": True})
    chunks.append(root)

    # Sections
    sec_boundaries: list[tuple[int, str]] = [(0, "début")]
    for m in _SECTION_RE.finditer(text):
        sec_boundaries.append((m.start(), m.group(0).strip()))
    sec_boundaries.append((len(text), "fin"))

    for sec_i, (sec_start, sec_label) in enumerate(sec_boundaries[:-1]):
        sec_end = sec_boundaries[sec_i + 1][0]
        sec_text = text[sec_start:sec_end].strip()
        if not sec_text:
            continue

        sec_id = _make_id(doc.document_id, "hierarchical", "section", str(sec_i))
        sec_chunk = _chunk_from(
            doc, sec_id, sec_text[:1200],
            parent_id=root_id,
            page_number=_page_of(sec_start, doc.pages),
            extra_meta={"level": "section", "section_label": sec_label},
        )
        chunks.append(sec_chunk)

        # Articles dans cette section
        art_boundaries: list[tuple[int, str | None]] = [(sec_start, None)]
        for m in _ARTICLE_RE.finditer(text, sec_start, sec_end):
            art_num = _article_number_from_match(m)
            art_boundaries.append((m.start(), art_num or None))
        art_boundaries.append((sec_end, None))

        for art_i, (art_start, art_num) in enumerate(art_boundaries[:-1]):
            art_end = art_boundaries[art_i + 1][0]
            art_text = text[art_start:art_end].strip()
            if not art_text:
                continue

            art_id = _make_id(doc.document_id, "hierarchical", "article", str(sec_i), str(art_i))
            art_chunk = _chunk_from(
                doc, art_id, art_text[:1200],
                parent_id=sec_id,
                article_number=art_num,
                page_number=_page_of(art_start, doc.pages),
                extra_meta={"level": "article"},
            )
            chunks.append(art_chunk)

            # Feuilles (paragraphes)
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", art_text) if p.strip()]
            for leaf_i, para in enumerate(paragraphs):
                if len(para) < 30:
                    continue
                # Subdivision si trop long
                sub_paras = [para[j:j + max_leaf_chars] for j in range(0, len(para), max_leaf_chars)]
                for sub_i, sub in enumerate(sub_paras):
                    leaf_id = _make_id(doc.document_id, "hierarchical", "leaf",
                                       str(sec_i), str(art_i), str(leaf_i), str(sub_i))
                    chunks.append(_chunk_from(
                        doc, leaf_id, sub,
                        parent_id=art_id,
                        article_number=art_num,
                        page_number=_page_of(art_start, doc.pages),
                        extra_meta={"level": "leaf"},
                    ))

    return chunks


# ---------------------------------------------------------------------------
# Stratégie 3 : Parent-Child Chunking
# ---------------------------------------------------------------------------

def chunk_parent_child(
    doc: RawDocument,
    *,
    parent_chars: int = 1200,
    child_chars: int = 400,
    child_overlap: int = 80,
) -> list[Chunk]:
    """Segments larges (parents) + segments fins chevauchants (enfants).

    L'enfant contient ``child_chars`` caractères avec un chevauchement de
    ``child_overlap`` par rapport à l'enfant précédent.  Cela permet une
    recherche fine sur les enfants tout en récupérant le contexte complet
    via le parent.
    """
    text = doc.full_text
    chunks: list[Chunk] = []

    # Découper en parents par paragraphe, en limitant à parent_chars
    raw_parents: list[tuple[int, str]] = []  # (offset, texte)
    paragraphs = re.split(r"\n{2,}", text)
    buf, buf_start = "", 0
    cursor = 0
    for para in paragraphs:
        if buf and len(buf) + len(para) + 1 > parent_chars:
            raw_parents.append((cursor - len(buf), buf.strip()))
            buf = para
        else:
            buf = (buf + "\n\n" + para).lstrip() if buf else para
        cursor += len(para) + 2  # +2 pour "\n\n"
    if buf.strip():
        raw_parents.append((cursor - len(buf), buf.strip()))

    for p_i, (p_offset, p_text) in enumerate(raw_parents):
        if not p_text:
            continue
        # Détection d'article dans le parent
        art_match = _ARTICLE_RE.search(p_text)
        art_num = _article_number_from_match(art_match) if art_match else None

        parent_id = _make_id(doc.document_id, "parent_child", "parent", str(p_i))
        parent_chunk = _chunk_from(
            doc, parent_id, p_text,
            article_number=art_num,
            page_number=_page_of(p_offset, doc.pages),
            extra_meta={"level": "parent", "parent_index": p_i},
        )
        chunks.append(parent_chunk)

        # Enfants chevauchants
        step = child_chars - child_overlap
        for c_i, start in enumerate(range(0, len(p_text), step)):
            child_text = p_text[start:start + child_chars].strip()
            if len(child_text) < 30:
                continue
            child_id = _make_id(doc.document_id, "parent_child", "child", str(p_i), str(c_i))
            chunks.append(_chunk_from(
                doc, child_id, child_text,
                parent_id=parent_id,
                article_number=art_num,
                page_number=_page_of(p_offset, doc.pages),
                extra_meta={"level": "child", "child_index": c_i, "overlap": child_overlap},
            ))

    return chunks


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_STRATEGY_MAP = {
    "structural":   chunk_structural,
    "hierarchical": chunk_hierarchical,
    "parent_child": chunk_parent_child,
}


def chunk_document(doc: RawDocument, strategy: str = "structural") -> list[Chunk]:
    """Découpe ``doc`` selon la stratégie demandée.

    Args:
        doc: Document brut extrait par :func:`~legal_rag.extraction.extract_document`.
        strategy: ``"structural"``, ``"hierarchical"`` ou ``"parent_child"``.

    Returns:
        Liste de :class:`~legal_rag.models.Chunk`.

    Raises:
        ValueError: Si la stratégie est inconnue.
    """
    fn = _STRATEGY_MAP.get(strategy)
    if fn is None:
        raise ValueError(f"Stratégie inconnue : {strategy!r}. Choisir parmi {list(_STRATEGY_MAP)}")
    return fn(doc)
