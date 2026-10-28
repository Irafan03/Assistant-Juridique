"""Construction du jeu de référence vérifié depuis l'index réel."""
import sys, pathlib, sqlite3, json
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"
OUT   = pathlib.Path(__file__).parent.parent / "evaluation" / "cases.json"

def find_chunks(conn, keywords, doc_type=None, limit=3):
    conds = ["instr(lower(text), ?) > 0" for _ in keywords]
    if doc_type:
        conds.append("document_type = ?")
    q = ("SELECT chunk_id, document_id, document_type, title, article_number, "
         "page_number, text FROM chunks WHERE " + " AND ".join(conds)
         + f" LIMIT {limit}")
    params = [k.lower() for k in keywords] + ([doc_type] if doc_type else [])
    return [dict(r) for r in conn.execute(q, params).fetchall()]

conn = sqlite3.connect(str(INDEX / "structural" / "metadata.db"))
conn.row_factory = sqlite3.Row

QUESTIONS = [
    {
        "id": "q_aml_01",
        "query": "Quelles sont les obligations d'une banque en matière de lutte contre le blanchiment d'argent ?",
        "note": "Circulaire 5/W/2017, loi 43-05 — obligations AML",
        "kw_sets": [
            (["blanchiment", "declaration", "soupcon"],    None),
            (["blanchiment", "vigilance", "client"],       None),
            (["blanchiment", "identification", "client"],  None),
            (["blanchiment", "conservation", "documents"], None),
            (["blanchiment", "procedure", "interne"],      None),
        ],
    },
    {
        "id": "q_fonds_01",
        "query": "Quelles règles prudentielles encadrent les fonds propres et la solvabilité des établissements de crédit ?",
        "note": "Loi 51-20 art.76, circulaire 14/G/2013",
        "kw_sets": [
            (["fonds propres", "solvabilite"],   None),
            (["fonds propres", "risques"],       "circulaire"),
            (["solvabilite", "liquidite"],       "loi"),
        ],
    },
    {
        "id": "q_aml_02",
        "query": "Quelles sont les obligations de déclaration de soupçon pour les établissements de crédit ?",
        "note": "Déclaration de soupçon — loi 43-05",
        "kw_sets": [
            (["declaration", "soupcon", "blanchiment"], None),
            (["soupcon", "etablissement", "credit"],    None),
        ],
    },
]

cases = []
print("Construction du jeu de référence AML + fonds propres\n")

for q in QUESTIONS:
    all_chunks = {}
    for kws, dt in q["kw_sets"]:
        rows = find_chunks(conn, kws, doc_type=dt, limit=3)
        for r in rows:
            if r["chunk_id"] not in all_chunks:
                all_chunks[r["chunk_id"]] = r

    chunk_ids = list(all_chunks.keys())
    circulaire_ids = [cid for cid, r in all_chunks.items() if r["document_type"] == "circulaire"]
    loi_ids        = [cid for cid, r in all_chunks.items() if r["document_type"] in ("loi", "texte")]

    print(f"[{q['id']}] {q['query'][:65]}")
    print(f"  {len(chunk_ids)} passages de référence trouvés "
          f"({len(circulaire_ids)} circulaires, {len(loi_ids)} lois)")
    for cid, r in list(all_chunks.items())[:3]:
        print(f"  → {r['document_type']} art={r['article_number']} | "
              f"{(r['title'] or '')[:50]}")
        print(f"    {r['text'][:80].replace(chr(10), ' ')}")

    cases.append({
        "id": q["id"],
        "query": q["query"],
        "note": q["note"],
        "relevant_circulars":  circulaire_ids,
        "relevant_laws":       loi_ids,
        "relevant_articles":   chunk_ids,
        "_verified_from_index": True,
        "_keywords_used": [kws for kws, _ in q["kw_sets"]],
    })

conn.close()

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False, indent=2)
print(f"\n{len(cases)} cas sauvegardés dans {OUT}")
print("Note : jugements extraits de l'index réel par mots-clés — vérification manuelle recommandée.")
