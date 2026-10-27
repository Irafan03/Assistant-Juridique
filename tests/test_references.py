from legal_rag.references import extract_references, normalize_reference


def test_extracts_explicit_references_in_french_and_arabic_context():
    refs = extract_references("Voir la loi n° 12-03 et les articles 4 et 5.")
    normalized = {normalize_reference(ref) for ref in refs}
    assert "loi n°12-03" in normalized
    assert "article 4 et 5" in normalized
