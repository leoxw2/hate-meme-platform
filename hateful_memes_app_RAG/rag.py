import hashlib
import chromadb
from chromadb.utils import embedding_functions

class RagRetriever:
    """kNN-Few-Shot-Retriever über gelabelte train-Memes.

    Jedes Dokument ist die Konkatenation aus Bildbeschreibung + Meme-Text
    (für das Embedding). Label/Text/Beschreibung liegen in den Metadaten,
    damit get_examples() formatierte Few-Shot-Beispiele liefern kann.
    """
    COLLECTION = "train_fewshot_zsrp"

    def __init__(self, db_path: str):
        self._client = chromadb.PersistentClient(path=db_path)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2")
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION, embedding_function=self._ef)

    @staticmethod
    def _embed_text(description: str, text: str) -> str:
        return f"{description} {text}".strip()

    def _id(self, doc: str) -> str:
        return hashlib.md5(doc.encode()).hexdigest()[:16]

    def add_examples(self, rows: list[dict]) -> None:
        """rows: list of {"label", "text", "description"}.

        Embedding-Dokument = description + " " + text. Idempotent über md5-id.
        """
        existing = set(self._col.get()["ids"])
        docs, ids, metas = [], [], []
        for r in rows:
            description = str(r.get("description", ""))
            text = str(r.get("text", ""))
            doc = self._embed_text(description, text)
            doc_id = self._id(doc)
            if not doc or doc_id in existing or doc_id in ids:
                continue
            docs.append(doc)
            ids.append(doc_id)
            metas.append({
                "label": int(r["label"]),
                "text": text,
                "description": description,
            })
        if docs:
            self._col.add(documents=docs, ids=ids, metadatas=metas)

    def get_examples(self, query: str, k: int = 4) -> list[dict]:
        """Gibt bis zu k ähnliche Beispiele zurück: [{label, text, description}]."""
        if self._col.count() == 0:
            return []
        n = min(k, self._col.count())
        res = self._col.query(query_texts=[query], n_results=n,
                              include=["metadatas"])
        return res["metadatas"][0]

    def count(self) -> int:
        return self._col.count()
