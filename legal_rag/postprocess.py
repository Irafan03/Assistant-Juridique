"""Post-traitement de la réponse de recherche.

Opère dans la couche de composition (``server.py``) sans modifier
``retrieval.py`` ni les index. Cinq responsabilités :

1. **Filtrer les en-têtes/pieds de page** : chunks qui ne contiennent que
   l'en-tête répété du recueil (739 fois dans l'index) — aucune valeur
   juridique, polluent le top-k et la génération.

2. **Filtrer les chunks trop courts** (titre seul, ligne isolée) : moins de
   MIN_CONTENT_LEN caractères de contenu réel après suppression de l'en-tête.

3. **Nettoyer les numéros d'article** : « Article 35254 » est un artefact de
   parsing (numéro de page concaténé). Tout numéro > 999 → None.

4. **Dédupliquer** par contenu normalisé (cross-document) : mêmes obligations
   copiées dans plusieurs versions d'un texte (ex. loi 43-05 versions 2007
   et 2021) → on garde le score le plus élevé.

5. **Re-classer** par pertinence juridique (chevauchement titre + texte avec
   les concepts étendus de la requête).
"""

from __future__ import annotations

import re
from dataclasses import replace
from collections import defaultdict

from .models import Candidate, Chunk, SearchResponse
from .answer import _fold, _key_terms, _overlap
from .concepts import expand_query_terms

# ── Constantes ────────────────────────────────────────────────────────────────

# Pattern d'en-tête répété dans le recueil BAM (vérifiable : 739 occurrences index)
_HEADER_RE = re.compile(
    r"RECUEIL DES TEXTES LEGISLATIFS"
    r"|REGISSANT L.ACTIVITE DES ETABLISSEMENTS"
    r"|ORGANISMES ASSIMILES"
    r"|MIS\s+À\s+JOUR\s+À\s+FIN"
    r"|Mis\s+à\s+jour\s+à\s+fin\s+20"
    r"|^\s*PUBLIC\s*$"
    r"|^\s*\d{1,4}[\xad\-]?\s*$",    # ligne = numéro de page seul (ex. "639­")
    re.I | re.MULTILINE
)

# Longueur minimale de contenu réel (après suppression des en-têtes)
MIN_CONTENT_LEN = 80

# Bornes article juridique plausible
_ART_MIN, _ART_MAX = 1, 999
_ART_VALID_RE = re.compile(r"^\d{1,3}(?:\s*(?:bis|ter|quater))?$", re.I)


# ── Filtrage des en-têtes ──────────────────────────────────────────────────────

def _strip_header_lines(text: str) -> str:
    """Supprime les lignes d'en-tête répétées du recueil, retourne le reste."""
    lines = text.splitlines()
    cleaned = [l for l in lines if not _HEADER_RE.search(l)]
    return "\n".join(cleaned).strip()


def is_header_chunk(chunk: Chunk) -> bool:
    """Retourne True si le chunk n'est qu'un en-tête ou pied de page sans
    contenu juridique exploitable.

    Critères :
    - Après suppression des lignes d'en-tête, il reste < MIN_CONTENT_LEN chars.
    - OU le contenu nettoyé < 30 % du texte brut (dominé par les en-têtes).
    - OU le texte brut contient l'en-tête ET le contenu nettoyé ne commence
      pas par une ligne d'obligation (le chunk n'a que l'en-tête + un titre).
    """
    text = (chunk.text or "").strip()
    if not text:
        return True
    cleaned = _strip_header_lines(text)
    # Trop court après nettoyage
    if len(cleaned) < MIN_CONTENT_LEN:
        return True
    # Dominé à > 70 % par les en-têtes
    if len(cleaned) < 0.30 * len(text):
        return True
    # Contient l'en-tête ET le premier vrai contenu est un titre de section
    # (tout en majuscules ou très court) → pas une obligation actionnable
    if _HEADER_RE.search(text):
        first_content_lines = [
            l.strip() for l in cleaned.splitlines()
            if l.strip() and not l.strip().isdigit()
        ]
        if first_content_lines:
            first = first_content_lines[0]
            # Titre de section tout en majuscules après nettoyage
            if first == first.upper() and len(first) < 120:
                return True
    return False


def is_content_too_short(chunk: Chunk) -> bool:
    """Retourne True si le chunk est un titre seul ou une ligne isolée."""
    text = (chunk.text or "").strip()
    cleaned = _strip_header_lines(text)
    return len(cleaned) < MIN_CONTENT_LEN


# ── Nettoyage numéro d'article ────────────────────────────────────────────────

def clean_article_number(art: str | None) -> str | None:
    """Retourne un numéro d'article fiable, ou None si non identifiable."""
    if art is None:
        return None
    a = str(art).strip()
    if not a:
        return None
    if _ART_VALID_RE.match(a):
        num = int(re.match(r"\d+", a).group())
        if _ART_MIN <= num <= _ART_MAX:
            return a
        return None  # > 999 : artefact
    # « 595-6 » : plage d'articles (Code de procédure pénale)
    if re.match(r"^\d{1,3}-\d{1,3}$", a):
        return a
    return None


# ── Déduplication ────────────────────────────────────────────────────────────

def _content_key(text: str, n: int = 150) -> str:
    """Clé de normalisation pour la déduplication cross-document."""
    return _fold(text or "")[:n].strip()


def deduplicate_passages(passages: list[Candidate]) -> list[Candidate]:
    """Déduplique par contenu normalisé, CROSS-DOCUMENT.

    Deux passages de documents différents avec le même texte (ex. article 31
    copié dans la loi 2007 et la loi 2021) ne comptent qu'une fois : on garde
    le score RRF le plus élevé.

    Ancienne logique (par (doc_id, texte)) gardait les deux versions —
    c'est la cause directe du doublon [1]=[5] observé.
    """
    best: dict[str, Candidate] = {}  # content_key → meilleur candidat
    order: list[str] = []            # pour conserver l'ordre d'apparition
    for cand in passages:
        key = _content_key(cand.chunk.text)
        if key not in best:
            order.append(key)
            best[key] = cand
        elif cand.score > best[key].score:
            best[key] = cand
    return [best[k] for k in order]


# ── Re-classement ─────────────────────────────────────────────────────────────

def relevance_score(chunk: Chunk, expanded_terms: list[str]) -> float:
    """Score de pertinence juridique pour le re-classement.

    Diagnostic 1c a montré que le boost titre ×3 ne discriminait pas (titres
    tronqués, peu de termes) et créait des faux positifs (titre de section
    « LUTTE CONTRE LE BLANCHIMENT » matchait tous les termes).

    Nouvelle pondération mesurée :
    - TITRE : poids ×2 UNIQUEMENT pour les références juridiques exactes
      (ex. « loi n°43-05 », « circulaire n°5/W/2017 ») — pas pour le texte
      générique. Détecté via un pattern de référence, pas un simple overlap.
    - TEXTE : poids ×3 (le corps de l'article contient les obligations réelles).
    - Les en-têtes de section (tout en majuscules) ne sont pas boostés.
    """
    title_hay = _fold(chunk.title or "")
    text_hay  = _fold(chunk.text  or "")

    # Score texte : nombre de termes étendus présents dans le corps
    text_hits = sum(1 for t in expanded_terms
                    if re.search(r"\b" + re.escape(t) + r"\b", text_hay))

    # Score titre : seulement pour les références juridiques exactes
    # (numéros de loi/circulaire/décret), pas pour le texte générique.
    title_ref_hits = 0
    for t in expanded_terms:
        # Une « référence juridique » contient un chiffre ou « n° »
        if re.search(r"\d|n\b", t) and re.search(r"\b" + re.escape(t) + r"\b", title_hay):
            title_ref_hits += 1

    # Pénalité : titre de section tout en majuscules (en-tête, faux positif)
    penalty = 0.0
    if chunk.text and chunk.text.strip():
        first_lines = [l.strip() for l in chunk.text.splitlines() if l.strip()][:3]
        for l in first_lines:
            if l == l.upper() and len(l) < 120:
                penalty += 1.0  # pénalise les en-têtes de section

    return text_hits * 3.0 + title_ref_hits * 2.0 - penalty


def rerank_passages(passages: list[Candidate], query: str) -> list[Candidate]:
    """Re-classe par pertinence juridique, tri stable sur l'ordre RRF.
    Re-numérote les citation_id en [1], [2], …
    """
    base_terms = _key_terms(query)
    expanded, _, _ = expand_query_terms(query, base_terms)
    scored = [(relevance_score(c.chunk, expanded), i, c)
              for i, c in enumerate(passages)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    reranked = [c for _, _, c in scored]
    return [replace(c, citation_id=f"[{rank}]") for rank, c in enumerate(reranked, 1)]


# ── Pipeline principal ────────────────────────────────────────────────────────

def filter_passages(passages: list[Candidate]) -> tuple[list[Candidate], list[str]]:
    """Filtre en-têtes et chunks trop courts. Retourne (gardés, raisons_rejetés)."""
    kept: list[Candidate] = []
    rejected: list[str] = []
    for cand in passages:
        if is_header_chunk(cand.chunk):
            rejected.append(f"en-tête: {(cand.chunk.text or '')[:60]!r}")
        elif is_content_too_short(cand.chunk):
            rejected.append(f"trop court ({len((cand.chunk.text or '').strip())} chars): "
                            f"{(cand.chunk.text or '')[:40]!r}")
        else:
            kept.append(cand)
    return kept, rejected


def process_response(response: SearchResponse, query: str) -> SearchResponse:
    """Pipeline complet de post-traitement :

    1. Nettoyage article_number (tous passages/catégories).
    2. Filtrage des en-têtes et chunks courts.
    3. Déduplication cross-document par contenu normalisé.
    4. Re-classement par pertinence juridique + re-numérotation des citations.
    5. Mise à jour des citation_id des catégories dérivées.
    6. Ajout d'un avertissement si des passages ont été filtrés.
    """
    if not response.passages:
        return response

    # ── 1. Nettoyage article_number ──────────────────────────────────────────
    def _clean_cand(c: Candidate) -> Candidate:
        new_art = clean_article_number(c.chunk.article_number)
        if new_art == c.chunk.article_number:
            return c
        new_chunk = replace(c.chunk, article_number=new_art)
        return replace(c, chunk=new_chunk)

    passages   = [_clean_cand(c) for c in response.passages]
    circulares = [_clean_cand(c) for c in response.circulares]
    associated = [_clean_cand(c) for c in response.associated_texts]
    articles   = [_clean_cand(c) for c in response.articles]

    # ── 2. Filtrage en-têtes / chunks courts ─────────────────────────────────
    passages, rejected = filter_passages(passages)

    # ── 3. Déduplication cross-document ──────────────────────────────────────
    passages = deduplicate_passages(passages)

    # ── 4. Re-classement ──────────────────────────────────────────────────────
    passages = rerank_passages(passages, query)

    # ── 5. Mise à jour des citation_id des catégories ─────────────────────────
    citation_by_chunk = {c.chunk.chunk_id: c.citation_id for c in passages}

    def _update_cite(items: list[Candidate]) -> list[Candidate]:
        return [replace(c, citation_id=citation_by_chunk.get(c.chunk.chunk_id))
                for c in items]

    circulares = _update_cite(circulares)
    associated = _update_cite(associated)
    articles   = _update_cite(articles)

    # ── 6. Avertissement si filtrages significatifs ────────────────────────────
    warnings = list(response.warnings)
    if rejected:
        warnings.append(
            f"{len(rejected)} passage(s) filtré(s) (en-têtes/trop courts) sur "
            f"{len(response.passages)} initiaux."
        )

    return SearchResponse(
        query=response.query,
        strategy=response.strategy,
        retrieval_mode=response.retrieval_mode,
        circulares=circulares,
        associated_texts=associated,
        articles=articles,
        passages=passages,
        answer=response.answer,
        warnings=warnings,
    )
