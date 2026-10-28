# Assistant juridique — RAG juridique (Prompt 1 + Prompt 2)

Ce dépôt contient les deux parties du moteur de recherche juridique :

- **Ingestion & indexation (Prompt 1)** — extraction PDF/DOCX, trois stratégies de chunking, embeddings locaux, stockage FAISS + SQLite.
- **Retrieval & API (Prompt 2)** — recherche hybride BM25 + vectorielle, reranking, API FastAPI, interface web.

---

## Architecture

```
legal_rag/
├── extraction.py   # Extraction PDF/DOCX avec OCR (Prompt 1)
├── chunking.py     # 3 stratégies de chunking (Prompt 1)
├── indexing.py     # Embeddings + FAISS/SQLite + IndexAdapter (Prompt 1)
├── models.py       # Schéma commun Chunk, Candidate, SearchResponse
├── interfaces.py   # Protocoles IndexAdapter, Reranker, AnswerGenerator
├── references.py   # Extraction de références juridiques FR/AR
├── retrieval.py    # Moteur hybride BM25 + RRF (Prompt 2)
├── metrics.py      # Precision@k, Recall@k, MRR, nDCG@k (Prompt 2)
├── evaluation.py   # Harnais d'évaluation sur 20 questions (Prompt 2)
└── api.py          # API FastAPI + interface web (Prompt 2)
```

---

## Partie ingestion — Prompt 1

### Installation

```powershell
python -m pip install -e ".[dev]"
# Optionnel : OCR (nécessite Tesseract installé sur le système)
# https://github.com/tesseract-ocr/tesseract
# Optionnel : GPU
# pip install faiss-gpu  (à la place de faiss-cpu)
```

> **Sans `faiss-cpu`** : un backend numpy de secours est activé automatiquement.
> Les performances ne sont pas optimisées pour de grands corpus ; installez `faiss-cpu` pour le corpus complet.

### Extraction de documents

```python
from legal_rag.extraction import extract_document

doc = extract_document("corpus/circulaire-5.pdf", ocr_lang="fra+ara")
print(doc.detected_type)   # "circulaire" — auto-détecté
print(doc.metadata["title"])
# Les clés auto-extraites sont listées dans doc.metadata["_auto_keys"]
```

- Supporte **PDF** (PyMuPDF) et **DOCX** (python-docx).
- Active l'**OCR** (pytesseract + Pillow) si une page contient moins de 50 caractères.
- Les champs `detected_type`, `title`, `date` sont marqués `_auto_keys` : non vérifiés.

### Trois stratégies de chunking

```python
from legal_rag.chunking import chunk_document

chunks_s  = chunk_document(doc, "structural")    # par articles
chunks_h  = chunk_document(doc, "hierarchical")  # arbre doc->section->article->feuille
chunks_pc = chunk_document(doc, "parent_child")  # larges parents + enfants chevauchants
```

Chaque chunk est un `Chunk` conforme au schéma commun :
`document_id`, `chunk_id`, `text`, `document_type`, `title`, `parent_id`,
`source_file`, `page_number`, `article_number`, `references`.

### Indexation et recherche

```python
from legal_rag.indexing import LegalIndex

# Construire les trois index
index = LegalIndex("data/index")
index.add_documents(
    ["corpus/circulaire-5.pdf", "corpus/loi-12-03.pdf"],
    show_progress=True,
)

# Chaque stratégie est indépendante
chunks = index.iter_chunks(strategy="structural")
results = index.search("règles de provisionnement", strategy="hierarchical", limit=5)
```

**Réindexation sans doublon** : un document déjà indexé (même `chunk_id`) est mis à jour
plutôt que dupliqué.

**Modèle d'embedding** : `paraphrase-multilingual-MiniLM-L12-v2` (multilingue FR/AR,
téléchargé automatiquement par `sentence-transformers`). Peut être remplacé :

```python
from legal_rag.indexing import EmbeddingModel, LegalIndex
index = LegalIndex("data/index", model_name="intfloat/multilingual-e5-base")
```

**Structure sur disque** :
```
data/index/
├── structural/    faiss.index  +  metadata.db
├── hierarchical/  faiss.index  +  metadata.db
└── parent_child/  faiss.index  +  metadata.db
```

### Connexion avec la partie Prompt 2

`LegalIndex` implémente directement le protocole `IndexAdapter` défini par Prompt 2 :

```python
from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.api import create_app

index = LegalIndex("data/index")
retriever = LegalRetriever(index)
app = create_app(retriever, document_root="corpus/")
# uvicorn my_module:app --reload
```

---

## Partie retrieval & API — Prompt 2

### Contrat avec le module d'indexation

Fournir un objet conforme à `legal_rag.interfaces.IndexAdapter` :

- `search(query, strategy, limit)` : recherche sémantique dans l'index de la stratégie indiquée.
- `iter_chunks(strategy)` : passages de cette stratégie, pour BM25 et les références exactes.
- `get_chunk(chunk_id, strategy)` et `get_parent(chunk, strategy)` : récupération des parents.

`strategy` vaut `structural`, `hierarchical` ou `parent_child`.

### Fonctionnement

- Recherche vectorielle fournie par l'adaptateur et BM25 local, combinés par Reciprocal Rank Fusion (RRF).
- Mode `semantic` pour la référence vectorielle seule et mode `hybrid` par défaut.
- Détection des références au format légal français et arabe, puis résolution exacte contre les références présentes dans les chunks.
- Les relations explicites sont étiquetées `explicit_reference` ; les voisins sémantiques restent `semantic_similarity`.
- Un reranker peut être fourni via `Reranker`. Aucun modèle externe n'est imposé.
- L'interface web est servie sur `/` ; API sur `POST /api/search`, originaux sur `GET /api/documents/{id}/file`.

### Requête API

```json
{"query": "Quelles règles la circulaire applique-t-elle au congé ?",
 "strategy": "structural", "retrieval_mode": "hybrid", "limit": 10, "generate_answer": false}
```

---

## Évaluation sur le corpus

Créer `evaluation/cases.json` avec exactement 20 questions annotées depuis les documents réels :

```json
[{
  "id": "q01",
  "query": "Question vérifiée sur un document réel",
  "relevant_circulars": ["chunk-circulaire-1"],
  "relevant_laws": ["chunk-loi-4"],
  "relevant_articles": ["chunk-article-17"]
}]
```

Les métriques (Precision@k, Recall@k, MRR, nDCG@k) sont calculées séparément pour
circulaires, textes associés et articles, par stratégie et par mode de recherche.

---

## Vérification

```powershell
python -m pytest
```

Les tests couvrent : extraction, chunking (3 stratégies), indexation, réindexation sans doublon,
métriques, références explicites, parents, textes associés et API.
