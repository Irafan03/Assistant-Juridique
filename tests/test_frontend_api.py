"""Tests d'intégration : frontend servi par FastAPI + routes statiques."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from legal_rag.api import create_app
from legal_rag.models import Chunk
from legal_rag.retrieval import LegalRetriever


FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


class MinimalIndex:
    chunk = Chunk(
        "doc-1", "chunk-1",
        "Les établissements de crédit sont soumis à agrément.",
        document_type="loi", title="Loi n°103-12", article_number="26",
        source_file="/corpus/loi-103-12.pdf",   # source_file requis par /api/documents/{id}
    )

    def search(self, query, *, strategy, limit):
        return [self.chunk]

    def iter_chunks(self, *, strategy):
        return [self.chunk]

    def get_chunk(self, chunk_id, *, strategy):
        return self.chunk if chunk_id == self.chunk.chunk_id else None

    def get_parent(self, chunk, *, strategy):
        return None


@pytest.fixture
def client():
    return TestClient(create_app(LegalRetriever(MinimalIndex())))


# ── Routes API ────────────────────────────────────────────────────────────────
class TestApiRoutes:
    def test_root_redirects_to_static_when_frontend_exists(self, client):
        """/ doit rediriger vers /static/index.html si frontend/ existe."""
        if not FRONTEND_DIR.is_dir():
            pytest.skip("Dossier frontend/ absent")
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/static/index.html"

    def test_static_index_served(self, client):
        """Le fichier index.html du frontend doit être servi."""
        if not FRONTEND_DIR.is_dir():
            pytest.skip("Dossier frontend/ absent")
        resp = client.get("/static/index.html")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Assistant Juridique" in resp.text

    def test_static_css_served(self, client):
        """Le CSS principal doit être servi."""
        if not FRONTEND_DIR.is_dir():
            pytest.skip("Dossier frontend/ absent")
        resp = client.get("/static/css/main.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]

    def test_static_js_config_served(self, client):
        """config.js doit être servi."""
        if not FRONTEND_DIR.is_dir():
            pytest.skip("Dossier frontend/ absent")
        resp = client.get("/static/js/config.js")
        assert resp.status_code == 200
        assert "API_BASE" in resp.text
        assert "STRATEGIES" in resp.text

    def test_static_js_api_no_hardcoded_secrets(self, client):
        """api.js ne doit pas contenir de token ou clé API (patterns base64/hex denses)."""
        if not FRONTEND_DIR.is_dir():
            pytest.skip("Dossier frontend/ absent")
        resp = client.get("/static/js/api.js")
        assert resp.status_code == 200
        text = resp.text
        import re
        # Pattern réaliste de clé : chaîne base64 ou hex de 32+ caractères SANS espaces
        # (les commentaires avec des mots collés sont exclus par le pattern strict)
        suspicious = re.findall(r'\b[A-Za-z0-9+/]{40,}={0,2}\b', text)
        assert len(suspicious) == 0, f"Potentielle clé encodée dans api.js : {suspicious[:2]}"

    def test_experiment_page_served(self, client):
        """La page expérimentation doit être servie."""
        if not FRONTEND_DIR.is_dir():
            pytest.skip("Dossier frontend/ absent")
        resp = client.get("/static/pages/experiment.html")
        assert resp.status_code == 200
        assert "Expérimentation" in resp.text

    def test_nonexistent_static_returns_404(self, client):
        """Un fichier statique inexistant doit retourner 404."""
        if not FRONTEND_DIR.is_dir():
            pytest.skip("Dossier frontend/ absent")
        resp = client.get("/static/nonexistent.js")
        assert resp.status_code == 404

    def test_api_search_still_works(self, client):
        """L'ajout du frontend ne casse pas POST /api/search."""
        resp = client.post("/api/search", json={
            "query": "agrément crédit", "strategy": "structural",
            "retrieval_mode": "semantic", "limit": 5,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "passages" in body
        assert "circulaires" in body
        assert "textes_associes" in body
        assert "warnings" in body

    def test_api_search_response_schema(self, client):
        """Vérifier les champs de chaque passage."""
        resp = client.post("/api/search", json={"query": "crédit agrément"})
        assert resp.status_code == 200
        body = resp.json()
        for key in ("passages", "circulaires", "textes_associes", "articles", "warnings"):
            assert key in body, f"Champ manquant : {key}"
        if body["passages"]:
            p = body["passages"][0]
            for field in ("document_id", "chunk_id", "text", "score", "relation"):
                assert field in p, f"Champ passage manquant : {field}"

    def test_api_search_invalid_strategy(self, client):
        """Une stratégie inconnue doit retourner 422."""
        resp = client.post("/api/search", json={"query": "test", "strategy": "invalide"})
        assert resp.status_code == 422

    def test_api_search_empty_query(self, client):
        """Une requête vide doit retourner 422."""
        resp = client.post("/api/search", json={"query": ""})
        assert resp.status_code == 422

    def test_api_search_unconfigured_returns_503(self):
        """Sans index configuré, /api/search retourne 503."""
        no_index_client = TestClient(create_app())
        resp = no_index_client.post("/api/search", json={"query": "test"})
        assert resp.status_code == 503

    def test_api_document_metadata(self, client):
        """GET /api/documents/{id} retourne document_id et source_file."""
        resp = client.get("/api/documents/doc-1?strategy=structural")
        assert resp.status_code == 200
        body = resp.json()
        assert body["document_id"] == "doc-1"
        assert "source_file" in body


# ── Contenu HTML ──────────────────────────────────────────────────────────────
class TestFrontendContent:
    def test_index_contains_required_ui_elements(self, client):
        if not FRONTEND_DIR.is_dir():
            pytest.skip("Dossier frontend/ absent")
        resp = client.get("/static/index.html")
        text = resp.text
        # Éléments UI critiques
        assert 'id="chat-input"'     in text
        assert 'id="btn-send"'       in text
        assert 'id="chat-strategy"'  in text
        assert 'id="chat-mode"'      in text
        # Valeurs exactes backend
        assert 'value="structural"'  in text
        assert 'value="hierarchical"'in text
        assert 'value="parent_child"'in text
        assert 'value="hybrid"'      in text
        assert 'value="semantic"'    in text

    def test_experiment_contains_compare_button(self, client):
        if not FRONTEND_DIR.is_dir():
            pytest.skip("Dossier frontend/ absent")
        resp = client.get("/static/pages/experiment.html")
        text = resp.text
        assert 'id="btn-compare-all"' in text
        assert 'id="exp-query"'       in text
        assert 'id="exp-results"'     in text

    def test_config_js_strategy_values_match_backend(self, client):
        """Les stratégies dans config.js doivent correspondre exactement au backend."""
        if not FRONTEND_DIR.is_dir():
            pytest.skip("Dossier frontend/ absent")
        resp = client.get("/static/js/config.js")
        text = resp.text
        from legal_rag.interfaces import STRATEGIES
        for strategy in STRATEGIES:
            assert f"'{strategy}'" in text, f"Stratégie {strategy!r} absente de config.js"

    def test_no_hardcoded_api_url_in_html(self, client):
        """Les HTML ne doivent pas contenir d'URL d'API en dur (hors localhost)."""
        if not FRONTEND_DIR.is_dir():
            pytest.skip("Dossier frontend/ absent")
        for page in ("/static/index.html", "/static/pages/experiment.html"):
            text = client.get(page).text
            # Pas d'URL externe dans le HTML (localhost est autorisé pour dev)
            assert "https://api." not in text
            assert "http://api."  not in text
