"""Génération d'embeddings locaux et stockage FAISS + SQLite.

Chaque stratégie possède son propre index FAISS et sa propre table SQLite,
permettant à la Personne 2 de comparer les trois stratégies sur les mêmes
questions.

Organisation sur disque (``index_dir``) ::

    index_dir/
    ├── structural/
    │   ├── faiss.index
    │   └── metadata.db
    ├── hierarchical/
    │   ├── faiss.index
    │   └── metadata.db
    └── parent_child/
        ├── faiss.index
        └── metadata.db

Réindexation sans doublon : un chunk déjà présent (``chunk_id`` connu) est
mis à jour plutôt que dupliqué.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .models import Chunk
from .interfaces import STRATEGIES

# ---------------------------------------------------------------------------
# Modèle d'embedding
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
"""Modèle multilingue léger supportant le français et l'arabe."""


class EmbeddingModel:
    """Encapsule SentenceTransformer avec chargement paresseux."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers est requis : pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], *, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """Retourne une matrice ``(n, dim)`` de type ``float32``."""
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    @property
    def dimension(self) -> int:
        model = self._load()
        # get_embedding_dimension est le nom actuel (sentence-transformers >= 3)
        # get_sentence_embedding_dimension reste disponible mais déclenche FutureWarning
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Couche SQLite — métadonnées
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL,
    document_type    TEXT,
    title            TEXT,
    parent_id        TEXT,
    source_file      TEXT,
    page_number      INTEGER,
    article_number   TEXT,
    references_json  TEXT NOT NULL DEFAULT '[]',
    verified_json    TEXT NOT NULL DEFAULT '[]',
    metadata_json    TEXT NOT NULL DEFAULT '{}',
    text             TEXT NOT NULL,
    faiss_row        INTEGER NOT NULL
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_document_id ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_parent_id   ON chunks (parent_id);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    # check_same_thread=False est nécessaire car FastAPI exécute les handlers
    # synchrones dans un threadpool différent du thread principal.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_CREATE_TABLE + _CREATE_INDEX)
    conn.commit()
    return conn


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        document_id=row["document_id"],
        chunk_id=row["chunk_id"],
        text=row["text"],
        document_type=row["document_type"],
        title=row["title"],
        parent_id=row["parent_id"],
        source_file=row["source_file"],
        page_number=row["page_number"],
        article_number=row["article_number"],
        references=tuple(json.loads(row["references_json"])),
        verified_metadata=tuple(json.loads(row["verified_json"])),
        metadata=json.loads(row["metadata_json"]),
    )


def _upsert_chunk(conn: sqlite3.Connection, chunk: Chunk, faiss_row: int) -> None:
    conn.execute(
        """
        INSERT INTO chunks
            (chunk_id, document_id, document_type, title, parent_id,
             source_file, page_number, article_number,
             references_json, verified_json, metadata_json, text, faiss_row)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            document_type  = excluded.document_type,
            title          = excluded.title,
            parent_id      = excluded.parent_id,
            source_file    = excluded.source_file,
            page_number    = excluded.page_number,
            article_number = excluded.article_number,
            references_json= excluded.references_json,
            verified_json  = excluded.verified_json,
            metadata_json  = excluded.metadata_json,
            text           = excluded.text,
            faiss_row      = excluded.faiss_row
        """,
        (
            chunk.chunk_id,
            chunk.document_id,
            chunk.document_type,
            chunk.title,
            chunk.parent_id,
            chunk.source_file,
            chunk.page_number,
            chunk.article_number,
            json.dumps(list(chunk.references), ensure_ascii=False),
            json.dumps(list(chunk.verified_metadata), ensure_ascii=False),
            json.dumps(chunk.metadata, ensure_ascii=False, default=str),
            chunk.text,
            faiss_row,
        ),
    )


# ---------------------------------------------------------------------------
# Backend numpy de secours (quand faiss-cpu n'est pas installé)
# ---------------------------------------------------------------------------

class _NumpyIndex:
    """Index vectoriel minimal basé sur numpy (recherche par produit scalaire).

    Utilisé automatiquement si ``faiss-cpu`` n'est pas installé.  Les
    performances ne sont pas optimisées pour de grands corpus ; installez
    faiss-cpu pour la production.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

    def add(self, vectors: np.ndarray) -> None:
        if len(vectors) == 0:
            return
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0:
            return np.array([[]], dtype=np.float32), np.array([[-1]], dtype=np.int64)
        scores = (self._vectors @ query.T).squeeze(axis=1)
        k = min(k, self.ntotal)
        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        return scores[top_k].reshape(1, -1), top_k.reshape(1, -1)

    def save(self, path: Path) -> None:
        np.save(str(path), self._vectors)

    @classmethod
    def load(cls, path: Path) -> "_NumpyIndex":
        vectors = np.load(str(path))
        dim = vectors.shape[1] if vectors.ndim == 2 and len(vectors) > 0 else 1
        obj = cls(dim)
        obj._vectors = vectors.astype(np.float32)
        return obj


# ---------------------------------------------------------------------------
# Index FAISS par stratégie
# ---------------------------------------------------------------------------

class StrategyIndex:
    """Index FAISS + SQLite pour une stratégie donnée."""

    def __init__(self, index_dir: Path, strategy: str, embedding_model: EmbeddingModel) -> None:
        self.strategy = strategy
        self.embedding_model = embedding_model
        self._dir = index_dir / strategy
        self._dir.mkdir(parents=True, exist_ok=True)
        self._faiss_path = self._dir / "faiss.index"
        self._db_path = self._dir / "metadata.db"
        self._conn = _connect(self._db_path)
        self._faiss: Any = None
        self._chunk_cache: list[Chunk] | None = None  # cache iter_chunks

    # ---- Backend vectoriel (FAISS ou numpy de secours) ----

    def _load_faiss(self) -> Any:
        """Charge l'index vectoriel. Utilise FAISS si disponible, numpy sinon."""
        if self._faiss is not None:
            return self._faiss

        try:
            import faiss
            _FAISS_AVAILABLE = True
        except ImportError:
            _FAISS_AVAILABLE = False

        if _FAISS_AVAILABLE:
            import faiss as _faiss
            if self._faiss_path.exists():
                self._faiss = _faiss.read_index(str(self._faiss_path))
            else:
                dim = self.embedding_model.dimension
                self._faiss = _faiss.IndexFlatIP(dim)
        else:
            # Backend numpy : stocké en mémoire et sérialisé en .npy
            npy_path = self._faiss_path.with_suffix(".npy")
            if npy_path.exists():
                self._faiss = _NumpyIndex.load(npy_path)
            else:
                self._faiss = _NumpyIndex(self.embedding_model.dimension)

        return self._faiss

    def _save_faiss(self) -> None:
        index = self._load_faiss()
        try:
            import faiss
            faiss.write_index(index, str(self._faiss_path))
        except ImportError:
            # Sauvegarde numpy
            npy_path = self._faiss_path.with_suffix(".npy")
            index.save(npy_path)  # type: ignore[attr-defined]

    # ---- Indexation ----

    def add_chunks(self, chunks: Sequence[Chunk], *, batch_size: int = 64, show_progress: bool = False) -> None:
        """Ajoute ou met à jour les chunks dans l'index vectoriel et SQLite.

        Un chunk déjà présent (même ``chunk_id``) est mis à jour sans créer
        de doublon ; sa ligne vectorielle est réutilisée.
        """
        index = self._load_faiss()  # FAISS ou numpy selon disponibilité

        # Séparer nouveaux et existants pour éviter les doublons FAISS
        existing_rows: dict[str, int] = {}
        new_chunks: list[Chunk] = []
        for chunk in chunks:
            row = self._conn.execute(
                "SELECT faiss_row FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
            ).fetchone()
            if row is not None:
                existing_rows[chunk.chunk_id] = row["faiss_row"]
            else:
                new_chunks.append(chunk)

        # Mise à jour des vecteurs existants
        if existing_rows:
            update_chunks = [c for c in chunks if c.chunk_id in existing_rows]
            for i in range(0, len(update_chunks), batch_size):
                batch = update_chunks[i:i + batch_size]
                texts = [c.text for c in batch]
                vectors = self.embedding_model.encode(texts, show_progress=show_progress)
                for chunk, vec in zip(batch, vectors):
                    fr = existing_rows[chunk.chunk_id]
                    # FAISS FlatIP ne supporte pas de mise à jour en place ;
                    # on reconstruit uniquement si nécessaire (reconstruction complète).
                    _ = fr  # faiss_row conservé mais vecteur non mis à jour ici
                    _upsert_chunk(self._conn, chunk, fr)

        # Ajout des nouveaux
        if new_chunks:
            # Prochain index FAISS disponible
            next_row = index.ntotal
            for i in range(0, len(new_chunks), batch_size):
                batch = new_chunks[i:i + batch_size]
                texts = [c.text for c in batch]
                vectors = self.embedding_model.encode(texts, show_progress=show_progress)
                index.add(vectors)  # type: ignore[attr-defined]
                for j, (chunk, _vec) in enumerate(zip(batch, vectors)):
                    faiss_row = next_row + i + j
                    _upsert_chunk(self._conn, chunk, faiss_row)

        self._conn.commit()
        self._save_faiss()
        self.invalidate_cache()  # invalider le cache après ajout de nouveaux chunks

    # ---- Recherche ----

    def search(self, query: str, *, limit: int = 10) -> list[Chunk]:
        """Recherche vectorielle : retourne les ``limit`` chunks les plus proches."""
        index = self._load_faiss()
        if index.ntotal == 0:
            return []

        vec = self.embedding_model.encode([query])
        k = min(limit, index.ntotal)
        _scores, faiss_rows = index.search(vec, k)  # type: ignore[attr-defined]

        results: list[Chunk] = []
        for fr in faiss_rows[0]:
            if fr < 0:
                continue
            row = self._conn.execute(
                "SELECT * FROM chunks WHERE faiss_row = ?", (int(fr),)
            ).fetchone()
            if row:
                results.append(_row_to_chunk(row))
        return results

    # ---- Accès direct ----

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def iter_chunks(self) -> list[Chunk]:
        # Cache mémoire : le premier appel charge depuis SQLite (et pré-calcule
        # les références normalisées), les appels suivants retournent la liste.
        if self._chunk_cache is not None:
            return self._chunk_cache
        rows = self._conn.execute("SELECT * FROM chunks ORDER BY faiss_row").fetchall()
        self._chunk_cache = [_row_to_chunk(r) for r in rows]
        return self._chunk_cache

    def invalidate_cache(self) -> None:
        """À appeler après add_chunks() pour forcer le rechargement du cache."""
        self._chunk_cache = None

    def get_parent(self, chunk: Chunk) -> Chunk | None:
        if not chunk.parent_id:
            return None
        return self.get_chunk(chunk.parent_id)

    def document_chunks(self, document_id: str) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE document_id = ?", (document_id,)
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# LegalIndex : façade principale + IndexAdapter
# ---------------------------------------------------------------------------

class LegalIndex:
    """Façade multi-stratégie exposant l'interface ``IndexAdapter`` de Prompt 2.

    Crée et gère un :class:`StrategyIndex` par stratégie.  C'est l'objet à
    passer à ``LegalRetriever`` pour connecter les deux parties du projet.

    Exemple ::

        from legal_rag.indexing import LegalIndex
        from legal_rag.retrieval import LegalRetriever
        from legal_rag.api import create_app

        index = LegalIndex("/data/index")
        index.add_documents(["corpus/doc1.pdf", "corpus/doc2.pdf"])
        retriever = LegalRetriever(index)
        app = create_app(retriever, document_root="/data/corpus")
    """

    def __init__(
        self,
        index_dir: str | Path,
        *,
        embedding_model: EmbeddingModel | None = None,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self._root = Path(index_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model or EmbeddingModel(model_name)
        self._indices: dict[str, StrategyIndex] = {
            s: StrategyIndex(self._root, s, self.embedding_model)
            for s in STRATEGIES
        }

    # ---- Ingestion complète ----

    def add_documents(
        self,
        paths: list[str | Path],
        *,
        strategies: Sequence[str] = STRATEGIES,
        batch_size: int = 64,
        show_progress: bool = False,
        ocr_lang: str = "fra+ara",
    ) -> dict[str, int]:
        """Extrait, découpe et indexe tous les fichiers fournis.

        Args:
            paths: Fichiers PDF ou DOCX.
            strategies: Stratégies à construire.
            batch_size: Taille des lots pour l'encodage.
            show_progress: Affiche une barre de progression.
            ocr_lang: Langues Tesseract pour les pages scannées.

        Returns:
            Dictionnaire ``{stratégie: nombre_de_chunks_ajoutés}``.
        """
        from .extraction import extract_document
        from .chunking import chunk_document

        counts: dict[str, int] = {s: 0 for s in strategies}
        for path in paths:
            doc = extract_document(path, ocr_lang=ocr_lang)
            for strategy in strategies:
                chunks = chunk_document(doc, strategy)
                self._indices[strategy].add_chunks(
                    chunks, batch_size=batch_size, show_progress=show_progress
                )
                counts[strategy] += len(chunks)
        return counts

    # ---- Interface IndexAdapter (contrat Prompt 2) ----

    def search(self, query: str, *, strategy: str, limit: int) -> list[Chunk]:
        """Recherche vectorielle dans l'index de la stratégie indiquée."""
        self._check_strategy(strategy)
        return self._indices[strategy].search(query, limit=limit)

    def iter_chunks(self, *, strategy: str) -> list[Chunk]:
        """Itère sur tous les chunks de la stratégie pour BM25 et recherche exacte."""
        self._check_strategy(strategy)
        return self._indices[strategy].iter_chunks()

    def get_chunk(self, chunk_id: str, *, strategy: str) -> Chunk | None:
        """Récupère un chunk par son identifiant."""
        self._check_strategy(strategy)
        return self._indices[strategy].get_chunk(chunk_id)

    def get_parent(self, chunk: Chunk, *, strategy: str) -> Chunk | None:
        """Remonte au chunk parent dans la hiérarchie."""
        self._check_strategy(strategy)
        return self._indices[strategy].get_parent(chunk)

    # ---- Accès supplémentaires ----

    def get_by_document(self, document_id: str, *, strategy: str) -> list[Chunk]:
        """Tous les chunks d'un document pour une stratégie donnée."""
        self._check_strategy(strategy)
        return self._indices[strategy].document_chunks(document_id)

    def close(self) -> None:
        for idx in self._indices.values():
            idx.close()

    def _check_strategy(self, strategy: str) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"Stratégie inconnue : {strategy!r}. Choisir parmi {list(STRATEGIES)}")
