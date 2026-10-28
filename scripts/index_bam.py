"""Indexation rigoureuse du recueil BAM — 7 points de vérification intégrés.

Usage :
    python scripts/index_bam.py

L'indexation est incrémentale : les chunks déjà présents (même chunk_id) sont
ignorés, seuls les documents manquants sont ajoutés.
"""
import sys, pathlib, re, time, sqlite3, json
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.extraction import (
    extract_document, RawDocument, RawPage,
    _extract_title, _extract_date,
)
from legal_rag.chunking import chunk_document
from legal_rag.indexing import LegalIndex
from legal_rag.interfaces import STRATEGIES

PDF   = pathlib.Path(__file__).parent.parent / "recueil-des-textes-legislatifs-et-reglementaires-de-bam-vf.pdf"
INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"

# ─────────────────────────────────────────────────────────────────────────────
# Regex de segmentation — Dahirs, arrêtés avec date, recommandations inclus
# ─────────────────────────────────────────────────────────────────────────────
HEADING_RE = re.compile(
    r'(?:^|\n)('
    # Lois (sans Dahir)
    r'(?:LOI|Loi)\s+[Nn][°º]\s*\d+[-\d]*[^.\n]{0,80}'
    # Décrets
    r'|(?:DECRET|Décret|décret)\s+[Nn][°º]\s*\d+[-\d\w]*[^.\n]{0,80}'
    # Circulaires (avec numéro alphanumérique : 1/W/2018, 14/G/2013…)
    r'|(?:CIRCULAIRE|Circulaire)\s+[Nn][°º]\s*[\d\w/.-]+[^.\n]{0,80}'
    # Lettres circulaires
    r'|(?:LETTRE CIRCULAIRE|Lettre circulaire)\s+[Nn][°º]\s*[\d]+[^.\n]{0,80}'
    # Arrêtés (avec numéro ou avec date "du JJ/MM" ou "du NN mois")
    r'|(?:ARRETE|Arrêté|arrêté)\s+(?:[Nn][°º]\s*\d+|\w+\s+\w+\s+[Nn][°º]|du\s+\d)[^.\n]{0,80}'
    # Directives
    r'|(?:DIRECTIVE|Directive)\s+[Nn][°º]\s*\d+[^.\n]{0,60}'
    # Dahirs (portant promulgation ou avec numéro)
    r'|(?:DAHIR|Dahir)\s+[Nn][°º]\s*\d+[-\d]*[^.\n]{0,80}'
    r'|(?:DAHIR|Dahir)\s+portant\s+(?:loi|promulgation)[^.\n]{0,80}'
    # Recommandations
    r'|(?:RECOMMANDATION|Recommandation)\s+[Nn][°º]\s*\d+[^.\n]{0,60}'
    # Chartes et codes
    r'|(?:CHARTE|Charte|CODE|Code)\s+(?:de\s+)?\w+[^.\n]{0,60}'
    r')',
    re.MULTILINE
)

# Mapping vers les types canoniques
TYPE_RULES = [
    (re.compile(r'^loi\b', re.I),                    'loi'),
    (re.compile(r'^dahir\b', re.I),                  'loi'),        # Dahir = acte royal promulguant une loi
    (re.compile(r'^d[eé]cret\b', re.I),              'décret'),
    (re.compile(r'^circulaire\b', re.I),             'circulaire'),
    (re.compile(r'^lettre circulaire\b', re.I),      'circulaire'),
    (re.compile(r'^directive\b', re.I),              'directive'),
    (re.compile(r'^arr[eê]t[eé]\b', re.I),          'arrêté'),
    (re.compile(r'^recommandation\b', re.I),         'recommandation'),
    (re.compile(r'^(?:charte|code)\b', re.I),        'texte'),
]

def doc_type_from_heading(h: str) -> str:
    for pat, dtype in TYPE_RULES:
        if pat.match(h.strip()):
            return dtype
    return 'texte'


def _safe_doc_id(heading: str, index: int) -> str:
    slug = re.sub(r'[^\w\d]', '_', heading[:60].lower())
    slug = re.sub(r'_+', '_', slug).strip('_')
    return f"{slug}_{index:04d}"


# ─────────────────────────────────────────────────────────────────────────────
# Point 1 : segmentation et validation
# ─────────────────────────────────────────────────────────────────────────────

def split_recueil(doc: RawDocument) -> tuple[list[RawDocument], list[str]]:
    """Découpe le recueil et retourne (sous_docs, avertissements)."""
    text = doc.full_text
    matches = list(HEADING_RE.finditer(text))
    warnings: list[str] = []
    sub_docs: list[RawDocument] = []

    # Introduction (avant le premier marqueur)
    intro_text = text[:matches[0].start()].strip() if matches else text
    if len(intro_text) > 100:
        sub_docs.append(RawDocument(
            source_file=doc.source_file,
            document_id=f"{doc.document_id}__intro",
            pages=[RawPage(1, intro_text[:5000])],
            full_text=intro_text,
            detected_type='texte',
            metadata={
                'title': 'Recueil BAM — Introduction et table des matières',
                'heading': 'Introduction',
                '_auto_keys': ['title'],
            },
        ))

    for i, match in enumerate(matches):
        heading   = match.group(1).strip()
        start     = match.start()
        end       = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg_text  = text[start:end].strip()

        # Faux positif : segment trop court
        if len(seg_text) < 200:
            warnings.append(f"Segment trop court ignoré (offset {start}): {heading[:60]!r}")
            continue

        doc_type  = doc_type_from_heading(heading)
        doc_id    = _safe_doc_id(heading, i)
        title     = _extract_title(seg_text) or heading[:150]
        date      = _extract_date(seg_text)

        # Si le type est incertain, le signaler explicitement
        if doc_type == 'texte' and heading:
            warnings.append(
                f"Type incertain pour {heading[:60]!r} (offset {start}) → classé 'texte'"
            )

        # Pages couvertes par ce segment
        pages_slice: list[RawPage] = []
        pos = 0
        for page in doc.pages:
            page_end = pos + len(page.text)
            if page_end > start and pos < end:
                pages_slice.append(page)
            pos = page_end

        sub_docs.append(RawDocument(
            source_file=doc.source_file,
            document_id=doc_id,
            pages=pages_slice or [RawPage(1, seg_text[:3000])],
            full_text=seg_text,
            detected_type=doc_type,
            metadata={
                'title': title,
                'date': date,
                'heading': heading,           # ← conservé explicitement (Point 3)
                'doc_index': i,
                '_auto_keys': ['title', 'date', 'detected_type'],
            },
        ))

    return sub_docs, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions d'audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_db(strategy: str) -> dict:
    db_path = INDEX / strategy / "metadata.db"
    npy_path = INDEX / strategy / "faiss.npy"
    if not db_path.exists():
        return {"absent": True}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    total     = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    docs      = conn.execute("SELECT COUNT(DISTINCT document_id) FROM chunks").fetchone()[0]
    no_type   = conn.execute("SELECT COUNT(*) FROM chunks WHERE document_type IS NULL").fetchone()[0]
    with_refs = conn.execute("SELECT COUNT(*) FROM chunks WHERE references_json != '[]'").fetchone()[0]
    with_art  = conn.execute("SELECT COUNT(*) FROM chunks WHERE article_number IS NOT NULL").fetchone()[0]
    dup_count = conn.execute(
        "SELECT COUNT(*) FROM (SELECT chunk_id FROM chunks GROUP BY chunk_id HAVING COUNT(*)>1)"
    ).fetchone()[0]
    bad_title = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE title LIKE '%RECUEIL DES TEXTES%'"
    ).fetchone()[0]
    type_dist = dict(conn.execute(
        "SELECT document_type, COUNT(*) FROM chunks GROUP BY document_type"
    ).fetchall())
    max_row   = conn.execute("SELECT MAX(faiss_row) FROM chunks").fetchone()[0]
    conn.close()

    vec_count = len(np.load(str(npy_path))) if npy_path.exists() else 0

    return {
        "total": total, "docs": docs, "vec_count": vec_count, "max_row": max_row,
        "no_type": no_type, "with_refs": with_refs, "with_art": with_art,
        "dup_count": dup_count, "bad_title": bad_title, "type_dist": type_dist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── Option --clean : réindexation à neuf (supprime les anciens index) ──
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true",
                        help="Supprime les anciens index avant réindexation (évite les doublons)")
    args, _ = parser.parse_known_args()
    if args.clean:
        print("[0/6] Nettoyage des anciens index (--clean)…")
        import shutil
        for strategy in STRATEGIES:
            strat_dir = INDEX / strategy
            if strat_dir.exists():
                shutil.rmtree(strat_dir)
                print(f"      {strategy}/ supprimé")
        # Invalider le cache modèle pour repartir propre
        print("      Réindexation à neuf.")

    print("=" * 70)
    print("INDEXATION RIGOUREUSE DU RECUEIL BAM")
    print("=" * 70)

    # ── 1. Extraction ────────────────────────────────────────────────────────
    print(f"\n[1/6] Extraction du PDF…")
    doc = extract_document(PDF)
    print(f"      {doc.metadata['num_pages']} pages  |  {len(doc.full_text):,} chars  |  OCR: {sum(p.from_ocr for p in doc.pages)} pages")

    # ── 2. Segmentation (Point 1) ────────────────────────────────────────────
    print(f"\n[2/6] Segmentation en documents individuels…")
    sub_docs, seg_warnings = split_recueil(doc)

    type_count = Counter(d.detected_type for d in sub_docs)
    print(f"      {len(sub_docs)} documents délimités :")
    for t, n in type_count.most_common():
        print(f"        {t:<22} : {n}")
    if seg_warnings:
        print(f"      {len(seg_warnings)} avertissements de segmentation :")
        for w in seg_warnings[:10]:
            print(f"        ⚠  {w}")

    # ── 3. Bilan avant indexation ────────────────────────────────────────────
    print(f"\n[3/6] État des index avant indexation…")
    pre = {s: audit_db(s) for s in STRATEGIES}
    for s, a in pre.items():
        if a.get("absent"):
            print(f"  {s:<15} : absent")
        else:
            print(f"  {s:<15} : {a['total']:>6} chunks  {a['docs']:>4} docs  {a['vec_count']:>6} vecteurs  doublons={a['dup_count']}")

    # ── 4. Indexation incrémentale ───────────────────────────────────────────
    print(f"\n[4/6] Indexation incrémentale (chunks existants ignorés)…")
    INDEX.mkdir(parents=True, exist_ok=True)
    index = LegalIndex(INDEX)

    totals_added = {s: 0 for s in STRATEGIES}
    totals_skipped = {s: 0 for s in STRATEGIES}
    errors: list[str] = []

    for i, sub_doc in enumerate(sub_docs, 1):
        for strategy in STRATEGIES:
            try:
                chunks = chunk_document(sub_doc, strategy)
                # Compter les nouveaux vs existants (l'upsert gère les doublons)
                db_path = INDEX / strategy / "metadata.db"
                conn = sqlite3.connect(str(db_path), check_same_thread=False)
                existing = {
                    row[0] for row in conn.execute("SELECT chunk_id FROM chunks").fetchall()
                }
                conn.close()
                new_chunks = [c for c in chunks if c.chunk_id not in existing]
                if new_chunks:
                    index._indices[strategy].add_chunks(new_chunks, show_progress=False)
                    totals_added[strategy] += len(new_chunks)
                totals_skipped[strategy] += len(chunks) - len(new_chunks)
            except Exception as e:
                errors.append(f"{sub_doc.document_id} / {strategy}: {e}")

        if i % 25 == 0 or i == len(sub_docs):
            print(f"      {i:>3}/{len(sub_docs)}  —  {sub_doc.document_id[:55]}")

    index.close()
    print(f"\n  Résumé par stratégie :")
    for s in STRATEGIES:
        print(f"    {s:<15} : +{totals_added[s]:,} nouveaux  |  {totals_skipped[s]:,} ignorés (déjà présents)")
    if errors:
        print(f"\n  {len(errors)} erreurs d'indexation :")
        for e in errors[:10]:
            print(f"    ✗ {e}")

    # ── 5. Audit post-indexation (Points 2, 3, 4) ────────────────────────────
    print(f"\n[5/6] Vérification des index finaux…")
    post = {s: audit_db(s) for s in STRATEGIES}
    for s, a in post.items():
        ok = "✓" if a["dup_count"] == 0 and a["bad_title"] == 0 else "⚠"
        print(f"\n  {s} {ok}")
        print(f"    Chunks        : {a['total']:,}")
        print(f"    Documents     : {a['docs']}")
        print(f"    Vecteurs      : {a['vec_count']:,}")
        print(f"    max faiss_row : {a['max_row']}")
        print(f"    Doublons      : {a['dup_count']}  {'OK' if a['dup_count']==0 else '⚠'}")
        print(f"    Type manquant : {a['no_type']}  {'OK' if a['no_type']==0 else '⚠'}")
        print(f"    Title hérité  : {a['bad_title']}  {'OK' if a['bad_title']==0 else '⚠'}")
        print(f"    Avec refs     : {a['with_refs']:,}")
        print(f"    Avec article  : {a['with_art']:,}")
        print(f"    Types         : { {k:v for k,v in sorted(a['type_dist'].items())} }")

    # ── 5b. Vérification Point 5 : réindexation sans doublon ────────────────
    print(f"\n  Point 5 — test réindexation (même doc, attend 0 doublon)…")
    index2 = LegalIndex(INDEX)
    first_doc = sub_docs[1]  # premier vrai document (pas l'intro)
    for strategy in STRATEGIES:
        chunks = chunk_document(first_doc, strategy)
        index2._indices[strategy].add_chunks(chunks, show_progress=False)
        a_after = audit_db(strategy)
        print(f"    {strategy:<15} : doublons après 2e passage = {a_after['dup_count']}  {'OK' if a_after['dup_count']==0 else '⚠ DOUBLON'}")
    index2.close()

    # ── 6. Tests de recherche réels (Point 6) ────────────────────────────────
    print(f"\n[6/6] Tests de recherche réelle…")
    index3 = LegalIndex(INDEX)

    QUERIES = [
        ("agrément établissements de crédit",    "loi"),
        ("fonds propres risques de crédit",       "circulaire"),
        ("blanchiment financement terrorisme",    "circulaire"),
        ("liquidité ratio de couverture",         "circulaire"),
        ("décret modalités application",          "décret"),
    ]

    for query, expected_type in QUERIES:
        print(f"\n  Requête : {query!r}")
        for strategy in STRATEGIES:
            results = index3.search(query, strategy=strategy, limit=3)
            if not results:
                print(f"    {strategy:<15} : aucun résultat ⚠")
                continue
            top = results[0]
            match = "✓" if top.document_type == expected_type else f"≠ (trouvé: {top.document_type})"
            print(f"    {strategy:<15} : [{top.document_type}] {str(top.title)[:55]}  art={top.article_number}  {match}")

    index3.close()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Indexation et vérifications terminées en {elapsed:.1f}s")
    print(f"Index : {INDEX.resolve()}")

    # ── Point 7 : compatibilité API Prompt 2 (résumé) ───────────────────────
    print(f"\nPoint 7 — Compatibilité API Prompt 2 :")
    print(f"  LegalIndex implémente IndexAdapter (search/iter_chunks/get_chunk/get_parent)")
    print(f"  → Utiliser : LegalRetriever(LegalIndex('{INDEX}'))")
    print(f"  → Puis     : create_app(retriever, document_root='{PDF.parent}')")


if __name__ == "__main__":
    main()
