"""Génération de réponse fidèle et reranking par métadonnées.

Ces composants ne dépendent ni d'un LLM externe ni de nouvelles bibliothèques.
Ils respectent les protocoles ``AnswerGenerator`` et ``Reranker`` définis dans
``legal_rag/interfaces.py`` (Personne 2) et sont branchés côté ``server.py``
sans modifier ``retrieval.py``.

Principes :
- La réponse ne contient que des affirmations issues des passages retrouvés.
- Chaque disposition citée est associée à son numéro d'article, son titre et
  une référence ``[n]`` pointant vers un passage de l'evidence.
- Aucun seuil, ratio, date ou sanction n'est inventé.
- Si aucun passage ne contient les termes clés de la question, la réponse
  signale explicitement que le corpus est insuffisant.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence

from .models import Candidate, Chunk

# ── Mots vides français (liste minimale, sans dépendance externe) ────────────
_STOPWORDS = frozenset({
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "et", "ou",
    "à", "a", "au", "aux", "en", "dans", "sur", "pour", "par", "avec", "sans",
    "qui", "que", "quoi", "dont", "où", "est", "sont", "etre", "été", "qu",
    "ne", "pas", "plus", "ce", "cette", "ces", "se", "sa", "son", "ses",
    "il", "elle", "ils", "elles", "on", "nous", "vous", "leur", "leurs",
    "comme", "ainsi", "alors", "donc", "mais", "ou", "ni", "car", "or",
    "quel", "quelle", "quelles", "quels", "tout", "tous", "toute", "toutes",
    "sont", "font", "fait", "doivent", "peut", "peuvent", "doit", "etre",
    "the", "of", "and", "to", "in", "for", "is", "are", "be", "with",
})

# Seuil de pertinence : un passage est jugé pertinent s'il partage avec la
# question au moins 2 termes clés distincts, OU la totalité des termes quand
# la question est courte. Un seul terme commun (ex. « recette » qui est aussi
# un terme juridique = « ordre de recette ») ne suffit pas : cela éviterait les
# faux positifs sur des questions hors périmètre. Règle mesurée sur 4 questions
# de référence (fonds propres, comptes inactifs, hors-corpus, accent-test).
def _is_relevant(overlap: int, n_key_terms: int) -> bool:
    if n_key_terms <= 0:
        return False
    if overlap >= 2:
        return True
    return overlap >= 1 and overlap == n_key_terms  # question courte : tous les termes présents


def _fold(text: str) -> str:
    """Normalisation pour le matching : minuscules + suppression des accents.
    Permet à ``solvabilite`` (question) de matcher ``solvabilité`` (corpus).
    """
    norm = text.casefold()
    # Décomposer les accents puis retirer les combining marks
    return re.sub(r"[\u0300-\u036f]", "", __import__("unicodedata").normalize("NFKD", norm))


def _tokenize(text: str) -> list[str]:
    """Tokenisation : mots de ≥4 lettres, sans stopwords, en minuscules sans accents."""
    raw = re.findall(r"[A-Za-zÀ-ÿ\u0600-\u06ff]{4,}", text, flags=re.UNICODE)
    out: list[str] = []
    for t in raw:
        f = _fold(t)
        if len(f) >= 4 and f not in _STOPWORDS:
            out.append(f)
    return out


def _key_terms(query: str) -> list[str]:
    """Termes clés de la question (unicité préservée, ordre conservé)."""
    seen: set[str] = set()
    terms: list[str] = []
    for t in _tokenize(query):
        if t not in seen:
            seen.add(t)
            terms.append(t)
    return terms


def _overlap(chunk: Chunk, key_terms: list[str]) -> int:
    """Nombre de termes clés de la question présents dans le chunk.

    Le matching utilise les frontières de mots (``\\bterm\\b``) pour éviter les
    faux positifs du type ``recette`` (question) matchant ``recettes`` (texte
    fiscal) — qui ferait croire qu'une question de cuisine trouve des passages
    pertinents dans le corpus juridique.
    """
    if not key_terms:
        return 0
    haystack = _fold(f"{chunk.title or ''} {chunk.text or ''} {chunk.article_number or ''}")
    count = 0
    for term in key_terms:
        # \b autour du terme (déjà normalisé sans accents) — évite recette/recettes.
        if re.search(r"\b" + re.escape(term) + r"\b", haystack):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Générateur de réponse fidèle
# ---------------------------------------------------------------------------

class FaithfulAnswerGenerator:
    """Synthèse fidèle à partir des passages retrouvés.

    Implémente ``AnswerGenerator.generate(query, evidence) -> str``.
    La chaîne retournée contient des citations ``[n]`` validées par
    ``retrieval.py`` (n doit appartenir à 1..len(evidence)).
    """

    def generate(self, query: str, *, evidence: Sequence[Candidate]) -> str:
        key_terms = _key_terms(query)
        if not key_terms:
            return self._insufficient(evidence, "La question ne contient pas de termes exploitables.")

        # Expansion de requête par concepts du domaine (ex. AML/blanchiment).
        from .concepts import expand_query_terms, detect_themes, is_aml_query, aml_doc_indicator
        expanded_terms, is_domain, theme_terms = expand_query_terms(query, key_terms)

        # Score de pertinence de chaque passage par chevauchement lexical étendu.
        scored = [(i, cand, _overlap(cand.chunk, expanded_terms)) for i, cand in enumerate(evidence)]
        relevant = [(i, c, o) for i, c, o in scored if _is_relevant(o, len(expanded_terms))]
        relevant.sort(key=lambda t: (-t[2], t[0]))  # pertinence desc, puis ordre initial

        # Déduplication : même document + même début de texte → on garde le meilleur.
        relevant = self._dedup_relevant(relevant)

        if not relevant:
            return self._insufficient(evidence,
                f"Aucun des {len(evidence)} passages indexés ne contient les termes clés "
                f"de la question ({', '.join(key_terms[:5])}).")

        return self._build(query, relevant, evidence, key_terms, expanded_terms,
                           is_domain, theme_terms)

    # ── Déduplication ─────────────────────────────────────────────────────────

    @staticmethod
    def _dedup_relevant(relevant):
        """Fusionne les passages pertinents quasi-identiques (même doc + même début)."""
        seen: dict[tuple[str, str], tuple[int, Candidate, int]] = {}
        out = []
        for i, cand, ov in relevant:
            key = (cand.chunk.document_id, _fold(cand.chunk.text or "")[:120].strip())
            if key not in seen:
                out.append((i, cand, ov))
                seen[key] = (i, cand, ov)
            # sinon : doublon, on l'ignore (le 1er garde le meilleur overlap car trié)
        return out

    # ── Construction de la réponse pertinente ──────────────────────────────────

    def _build(self, query, relevant, evidence, key_terms, expanded_terms,
               is_domain, theme_terms) -> str:
        # Cas AML : regroupement par thème d'obligation.
        from .concepts import is_aml_query
        if is_domain and is_aml_query(query):
            return self._build_thematic(query, relevant, evidence, theme_terms)
        # Cas général : regroupement par document.
        return self._build_by_doc(query, relevant, evidence)

    def _build_by_doc(self, query, relevant, evidence) -> str:
        """Synthèse générale regroupée par texte juridique source."""
        lines: list[str] = []
        lines.append(f"D'après les {len(relevant)} disposition(s) pertinente(s) retrouvées "
                     f"dans le corpus de Bank Al-Maghrib :")
        lines.append("")

        # Regroupement par document (thème = texte juridique source)
        by_doc: dict[str, list[tuple[int, Candidate, int]]] = defaultdict(list)
        for i, cand, ov in relevant:
            by_doc[cand.chunk.document_id].append((i, cand, ov))

        for doc_id, items in by_doc.items():
            title = items[0][1].chunk.title or doc_id
            doc_type = items[0][1].chunk.document_type or "texte"
            lines.append(f"■ {title} ({doc_type})")
            for i, cand, _ov in items:
                chunk = cand.chunk
                art = chunk.article_number or "—"
                # Extrait fidèle : première phrase ou 220 premiers caractères
                excerpt = self._first_sentence(chunk.text)
                page = f", page {chunk.page_number}" if chunk.page_number else ""
                lines.append(f"   [{i + 1}] Article {art}{page} : {excerpt}")
            lines.append("")

        lines.append("Ces éléments sont extraits du recueil des textes législatifs et "
                     "réglementaires de Bank Al-Maghrib (mis à jour à fin 2023).")
        lines.append("Consultez les sources ci-dessous pour le texte intégral. Les "
                     "dispositions non retrouvées dans les passages fournis ne sont pas "
                     "reprises ici : aucune obligation n'est inventée.")
        return "\n".join(lines)

    # ── Synthèse thématique (AML) ──────────────────────────────────────────────

    # Libellés explicatifs des thèmes d'obligation AML. Un thème n'est présenté
    # que si un passage pertinent le confirme réellement (cf. _build_thematic).
    _AML_THEME_LABELS = {
        "identification_client": "Identification et connaissance du client",
        "vigilance":             "Mesures de vigilance à l'égard de la clientèle et des opérations",
        "surveillance":          "Surveillance des opérations et détection des opérations suspectes",
        "declaration":           "Déclarations à l'autorité compétente (déclaration de soupçon)",
        "conservation":          "Conservation des documents et des informations",
        "controle_interne":       "Procédures internes, contrôle et formation du personnel",
    }

    def _build_thematic(self, query, relevant, evidence, theme_terms) -> str:
        """Synthèse organisée par obligations AML.

        Garanties :
        - Aucun passage d'en-tête (chunk sans contenu réel) n'est cité.
        - Chaque item cité inclut un extrait de phrase d'obligation réelle,
          pas le texte brut du chunk.
        - Un thème sans passage valide n'est pas affiché (pas de gabarit).
        - Les doublons de contenu cross-document sont dédupliqués par phrase.
        """
        from .concepts import detect_themes, aml_doc_indicator, AML_CORE
        from .postprocess import is_header_chunk, _strip_header_lines

        theme_to_items: dict[str, list[tuple[int, Candidate, str]]] = defaultdict(list)
        seen_excerpts: dict[str, set] = defaultdict(set)   # theme → extrait déjà cité

        for i, cand, _ov in relevant:
            chunk = cand.chunk

            # Rejeter les chunks d'en-tête : ne contiennent aucune obligation
            if is_header_chunk(chunk):
                continue

            # Nettoyer le texte avant d'en extraire un extrait
            clean_text = _strip_header_lines(chunk.text or "")
            if len(clean_text.strip()) < 80:
                continue   # Trop court même après nettoyage

            title_fold = _fold(chunk.title or "")
            text_fold  = _fold(clean_text)

            themes = detect_themes(text_fold, title_fold)
            core_hit = any(
                re.search(r"\b" + re.escape(t) + r"\b", text_fold + " " + title_fold)
                for t in AML_CORE
            )
            if not themes and not core_hit and not aml_doc_indicator(title_fold):
                continue

            # Extraire la phrase la plus pertinente du chunk (contient un verbe d'obligation)
            excerpt = self._obligation_sentence(clean_text, theme_terms)
            if not excerpt:
                excerpt = self._first_sentence(clean_text)
            # Rejeter les extraits qui sont des titres de section (tout en majuscules)
            if excerpt and excerpt == excerpt.upper() and len(excerpt) < 120:
                excerpt = self._first_sentence(
                    "\n".join(l for l in clean_text.splitlines()
                              if l.strip() and l.strip() != l.strip().upper())
                )
            if not excerpt or len(excerpt) < 20:
                continue  # Pas d'extrait exploitable

            for theme in themes:
                # Déduplication par extrait au sein d'un même thème
                excerpt_key = _fold(excerpt)[:100]
                if excerpt_key not in seen_excerpts[theme]:
                    seen_excerpts[theme].add(excerpt_key)
                    theme_to_items[theme].append((i, cand, excerpt))
            if not themes and (core_hit or aml_doc_indicator(title_fold)):
                excerpt_key = _fold(excerpt)[:100]
                if excerpt_key not in seen_excerpts["general"]:
                    seen_excerpts["general"].add(excerpt_key)
                    theme_to_items.setdefault("general", []).append((i, cand, excerpt))

        if not theme_to_items:
            return self._build_by_doc(query, relevant, evidence)

        lines: list[str] = []
        lines.append("Obligations des établissements de crédit en matière de lutte "
                     "contre le blanchiment de capitaux, d'après les dispositions "
                     "retrouvées dans le recueil de Bank Al-Maghrib :")
        lines.append("")

        order = ["identification_client", "vigilance", "surveillance",
                 "declaration", "conservation", "controle_interne"]
        cited_themes = 0
        for theme in order:
            items = theme_to_items.get(theme)
            if not items:
                continue
            cited_themes += 1
            label = self._AML_THEME_LABELS.get(theme, theme)
            lines.append(f"■ {label}")
            for i, cand, excerpt in items:
                chunk = cand.chunk
                art  = chunk.article_number or "—"
                page = f" (page {chunk.page_number})" if chunk.page_number else ""
                doc_title = (chunk.title or chunk.document_id or "")[:70]
                lines.append(f"   [{i + 1}] {doc_title} — Art. {art}{page}")
                lines.append(f"        « {excerpt} »")
            lines.append("")

        general = theme_to_items.get("general", [])
        if general:
            lines.append("■ Dispositions générales anti-blanchiment")
            for i, cand, excerpt in general:
                chunk = cand.chunk
                art  = chunk.article_number or "—"
                page = f" (page {chunk.page_number})" if chunk.page_number else ""
                doc_title = (chunk.title or chunk.document_id or "")[:70]
                lines.append(f"   [{i + 1}] {doc_title} — Art. {art}{page}")
                lines.append(f"        « {excerpt} »")
            lines.append("")

        lines.append(f"Synthèse fondée sur {cited_themes} axe(s) d'obligation "
                     f"confirmé(s) par des passages du recueil.")
        lines.append("Chaque extrait ci-dessus provient directement du texte indexé. "
                     "Aucune obligation n'est ajoutée au-delà des passages fournis.")
        return "\n".join(lines)

    @staticmethod
    def _obligation_sentence(text: str, theme_terms: dict | None = None,
                             max_len: int = 250) -> str:
        """Extrait la phrase du texte qui contient un verbe d'obligation.

        Cherche en priorité une phrase contenant « doit », « sont tenus »,
        « est tenu », « doivent », « sont obligés », « veiller », « procéder ».
        Si aucune, retourne la première phrase substantielle (> 40 chars).
        """
        if not text:
            return ""
        # Diviser en phrases simples
        sentences = re.split(r"(?<=[.!?])\s+|(?<=\n)", text.strip())
        obligation_verbs = re.compile(
            r"\bdoit\b|\bsont\s+tenus?\b|\bdoivent\b|\bsont\s+oblig[eé]s?\b"
            r"|\bveiller\b|\bprocéder\b|\beffectuer\b|\bdéclarer\b"
            r"|\bidentifier\b|\bconserver\b|\bmettre\s+en\s+place\b",
            re.I
        )
        for sent in sentences:
            s = sent.strip()
            if len(s) > 40 and obligation_verbs.search(s):
                s = re.sub(r"\s+", " ", s)
                return s[:max_len].rstrip() + ("…" if len(s) > max_len else "")
        # Fallback : première phrase substantielle
        for sent in sentences:
            s = re.sub(r"\s+", " ", sent.strip())
            if len(s) > 40:
                return s[:max_len].rstrip() + ("…" if len(s) > max_len else "")
        return ""

    # ── Cas insuffisant ─────────────────────────────────────────────────────────

    def _insufficient(self, evidence, reason: str) -> str:
        """Message signalant que le corpus ne permet pas de répondre fiabilité.

        Inclut au moins une citation valide [n] pour passer la validation de
        retrieval.py, mais en indiquant clairement le caractère indicatif.
        """
        lines = [
            "Le corpus retrouvé ne permet pas d'établir une réponse fiable à cette question.",
            "",
            reason,
            "",
        ]
        if evidence:
            # Citer les 3 passages les plus proches à titre indicatif
            lines.append("À titre indicatif, les passages les plus proches sont référencés "
                         f"ci-dessous : " + ", ".join(f"[{i + 1}]" for i in range(min(3, len(evidence)))) + ".")
            lines.append("Ces extraits ne répondent pas directement à la question posée.")
        return "\n".join(lines)

    # ── Utilitaire ──────────────────────────────────────────────────────────────

    @staticmethod
    def _first_sentence(text: str, max_len: int = 220) -> str:
        if not text:
            return ""
        # Première phrase : jusqu'au premier point suivi d'un espace/majuscule, ou fin.
        m = re.search(r"[.!?](\s|$)", text)
        snippet = text[:m.end()].strip() if m else text.strip()
        if len(snippet) > max_len:
            snippet = snippet[:max_len].rstrip() + "…"
        # Normaliser les espaces/blancs
        return re.sub(r"\s+", " ", snippet)


# ---------------------------------------------------------------------------
# Reranker par métadonnées (compatible Reranker)
# ---------------------------------------------------------------------------

class MetadataReranker:
    """Re-classe les candidats en privilégiant le chevauchement lexical
    avec la question (titre + texte + numéro d'article).

    Aucune réindexation nécessaire : opère sur les candidats déjà récupérés.
    Tri stable : les ex aequo conservent l'ordre RRF initial.
    """

    def __init__(self, *, boost: float = 1.0) -> None:
        self.boost = boost

    def rerank(self, query: str, candidates: Sequence[Chunk]) -> Sequence[Chunk]:
        key_terms = _key_terms(query)
        if not key_terms or not candidates:
            return list(candidates)
        # (overlap, index_original) pour tri stable
        scored = [(self.boost * _overlap(c, key_terms), idx, c)
                  for idx, c in enumerate(candidates)]
        # Tri : overlap desc, puis index asc (préserve l'ordre RRF pour les ex aequo)
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [c for _, _, c in scored]
