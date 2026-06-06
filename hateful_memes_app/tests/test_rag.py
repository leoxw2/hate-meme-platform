import tempfile
from rag import RagRetriever

def test_add_and_query():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        r = RagRetriever(d)
        r.add_documents(["Hate speech targets race.", "Religious discrimination.", "Gender stereotypes."])
        results = r.get_context("racist content", n_results=2)
        assert len(results) == 2 and isinstance(results[0], str)

def test_add_idempotent():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        r = RagRetriever(d)
        r.add_documents(["Same doc."])
        r.add_documents(["Same doc."])
        assert r.count() == 1

def test_empty_returns_empty():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        assert RagRetriever(d).get_context("anything") == []
