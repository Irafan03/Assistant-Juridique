"""Expansion de requête par concepts juridiques.

Le RAG lexical ne comprend pas l'intention juridique d'une question. Une
question sur le « blanchiment d'argent » doit aussi retrouver les dispositions
sur la vigilance, la connaissance du client, la déclaration de soupçon, etc.

Ce module fournit une carte de concepts limitée et vérifiée (terminologie
effectivement employée par le recueil de Bank Al-Maghrib) et des helpers
d'expansion utilisés par le générateur de réponse et le post-traitement.

La carte est volontairement restreinte au domaine AML/anti-blanchiment du
recueil. Tout autre domaine n'est pas étendu (pas d'invention de synonymes).
"""

from __future__ import annotations

import re
from .answer import _fold

# ── Domaine AML / lutte contre le blanchiment de capitaux ────────────────────
# Termes effectivement présents dans le recueil BAM (loi 43-05, circulaire
# 5/W/2017, code pénal, loi 12-18). Chaque thème regroupe les termes qui
# désignent la même obligation.

AML_THEMES: dict[str, list[str]] = {
    "identification_client": [
        "identification", "identite", "connaissance du client", "client",
        "clients", "beneficiaire effectif", "beneficiaires effectifs",
        "due diligence", "diligence",
    ],
    "vigilance": [
        "vigilance", "vigilance ordinaire", "vigilance renforcee",
        "mesures de vigilance",
    ],
    "surveillance": [
        "surveillance", "operations suspectes", "operation suspecte",
        "detection", "suivi", "monitoring", "examen approfondi",
    ],
    "declaration": [
        "declaration", "declarations", "declaration de soupcon",
        "declarations de soupcons", "soupcon", "soupcons", "tracfin",
        "cellule de renseignement financier", "autorite competente",
        "signalisation", "declarant", "declarants",
    ],
    "conservation": [
        "conservation", "archivage", "registre", "registres",
        "copies", "pieces", "periode de conservation",
    ],
    "controle_interne": [
        "controle interne", "procedure interne", "procedures internes",
        "formation", "personnel", "conformite", "officier de conformite",
        "organe de controle",
    ],
}

# Termes « cœur » : leur présence indique le contexte AML.
AML_CORE: list[str] = [
    "blanchiment", "capitaux", "financement du terrorisme",
    "anti-blanchiment", "lutte contre",
]

# Indicateurs AML dans le titre d'un document (pour confirmer le contexte).
AML_DOC_INDICATORS: list[str] = [
    "blanchiment", "43-05", "5/w/2017", "12-18", "financement du terrorisme",
]


def _matches_any(haystack_folded: str, terms: list[str]) -> list[str]:
    """Retourne les termes présents (word-boundary) dans le texte normalisé."""
    found = []
    for t in terms:
        if re.search(r"\b" + re.escape(t) + r"\b", haystack_folded):
            found.append(t)
    return found


def is_aml_query(query: str) -> bool:
    """Détecte si la question porte sur le blanchiment / AML."""
    q = _fold(query)
    return bool(_matches_any(q, AML_CORE + ["blanchiment d argent", "blanchiment de capitaux"]))


def detect_themes(text_folded: str, doc_title_folded: str = "") -> list[str]:
    """Thèmes AML présents dans un passage (texte + titre du document)."""
    hay = f"{text_folded} {doc_title_folded}"
    return [theme for theme, terms in AML_THEMES.items() if _matches_any(hay, terms)]


def expand_query_terms(query: str, base_terms: list[str]) -> tuple[list[str], bool, dict[str, list[str]]]:
    """Étend les termes de la requête avec les concepts du domaine détecté.

    Returns:
        expanded_terms : termes étendus (base + concepts du domaine)
        is_domain      : True si la requête relève d'un domaine étendu
        theme_terms    : {thème: [termes]} pour le regroupement de la réponse
    """
    if is_aml_query(query):
        extra: list[str] = []
        for terms in AML_THEMES.values():
            extra.extend(terms)
        # Dédupliquer en préservant l'ordre
        seen = set(base_terms)
        expanded = list(base_terms)
        for t in extra:
            if t not in seen:
                seen.add(t)
                expanded.append(t)
        return expanded, True, AML_THEMES
    return list(base_terms), False, {}


def aml_doc_indicator(title_folded: str) -> bool:
    """Indique si le titre du document évoque le domaine AML."""
    return bool(_matches_any(title_folded, AML_DOC_INDICATORS))
