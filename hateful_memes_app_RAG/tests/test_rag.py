import tempfile
from rag import RagRetriever

EXAMPLES = [
    {"label": 1, "text": "go back to where you came from",
     "description": "People in traditional minority clothing at a border."},
    {"label": 0, "text": "when you finally finish your homework",
     "description": "A child smiling and celebrating at a desk."},
    {"label": 1, "text": "look how many lives they ruined",
     "description": "A plate of a religious group's traditional food."},
]

def test_add_and_query():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        r = RagRetriever(d)
        r.add_examples(EXAMPLES)
        results = r.get_examples("racist xenophobic content", k=2)
        assert len(results) == 2
        assert all("label" in ex and "text" in ex and "description" in ex
                   for ex in results)

def test_add_idempotent():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        r = RagRetriever(d)
        r.add_examples([EXAMPLES[0]])
        r.add_examples([EXAMPLES[0]])
        assert r.count() == 1

def test_empty_returns_empty():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        assert RagRetriever(d).get_examples("anything") == []
