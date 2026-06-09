import hashlib
import chromadb
from chromadb.utils import embedding_functions

class RagRetriever:
    COLLECTION = "hate_speech_knowledge"

    def __init__(self, db_path: str):
        self._client = chromadb.PersistentClient(path=db_path)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2")
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION, embedding_function=self._ef)

    def _id(self, doc: str) -> str:
        return hashlib.md5(doc.encode()).hexdigest()[:16]

    def add_documents(self, docs: list[str]) -> None:
        existing = set(self._col.get()["ids"])
        new = [(d, self._id(d)) for d in docs if self._id(d) not in existing]
        if new:
            self._col.add(documents=[d for d, _ in new], ids=[i for _, i in new])

    def get_context(self, query: str, n_results: int = 3) -> list[str]:
        if self._col.count() == 0:
            return []
        n = min(n_results, self._col.count())
        return self._col.query(query_texts=[query], n_results=n)["documents"][0]

    def count(self) -> int:
        return self._col.count()
