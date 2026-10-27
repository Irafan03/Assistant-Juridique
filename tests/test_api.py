from fastapi.testclient import TestClient

from legal_rag.api import create_app
from legal_rag.models import Chunk
from legal_rag.retrieval import LegalRetriever


class OneChunkIndex:
    chunk = Chunk("doc-1", "chunk-1", "La circulaire décrit la procédure.", document_type="circulaire", title="Circulaire test")

    def search(self, query, *, strategy, limit):
        return [self.chunk]

    def iter_chunks(self, *, strategy):
        return [self.chunk]

    def get_chunk(self, chunk_id, *, strategy):
        return self.chunk if chunk_id == self.chunk.chunk_id else None

    def get_parent(self, chunk, *, strategy):
        return None


def test_search_api_returns_grouped_results_and_strategy():
    client = TestClient(create_app(LegalRetriever(OneChunkIndex())))
    response = client.post("/api/search", json={"query": "procédure", "strategy": "structural", "retrieval_mode": "semantic"})
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "structural"
    assert body["retrieval_mode"] == "semantic"
    assert body["circulaires"][0]["document_id"] == "doc-1"
    assert body["passages"][0]["citation_id"] == "[1]"


def test_unconfigured_app_reports_missing_index():
    response = TestClient(create_app()).post("/api/search", json={"query": "question"})
    assert response.status_code == 503
