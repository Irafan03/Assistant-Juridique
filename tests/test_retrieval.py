from legal_rag.models import Chunk
from legal_rag.retrieval import LegalRetriever


class MemoryIndex:
    def __init__(self, chunks):
        self.chunks = chunks

    def iter_chunks(self, *, strategy):
        return self.chunks

    def search(self, query, *, strategy, limit):
        return self.chunks[:limit]

    def get_chunk(self, chunk_id, *, strategy):
        return next((c for c in self.chunks if c.chunk_id == chunk_id), None)

    def get_parent(self, chunk, *, strategy):
        return self.get_chunk(chunk.parent_id, strategy=strategy) if chunk.parent_id else None


def test_exact_reference_and_parent_are_preserved():
    parent = Chunk("d1", "parent", "Circulaire sur le congé.", document_type="circulaire", references=("Loi n° 12-03",))
    child = Chunk("d1", "child", "Article 4 applique la loi n° 12-03.", document_type="circulaire", parent_id="parent", article_number="4")
    law = Chunk("d2", "law", "Article 1. Objet.", document_type="loi", title="Loi n° 12-03", references=("Loi n° 12-03",), article_number="1")
    result = LegalRetriever(MemoryIndex([parent, child, law])).search("Loi n° 12-03", limit=10)
    assert any(item.chunk.chunk_id == "parent" for item in result.passages)
    assert any(item.chunk.chunk_id == "law" for item in result.associated_texts)
    assert any(item.exact_reference for item in result.passages)


def test_empty_and_unknown_strategy_are_rejected():
    engine = LegalRetriever(MemoryIndex([]))
    try:
        engine.search(" ")
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        engine.search("test", strategy="missing")
        assert False, "expected ValueError"
    except ValueError:
        pass
