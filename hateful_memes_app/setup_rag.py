"""Einmalig ausführen: befüllt ChromaDB.
Dokumente auf Englisch — all-MiniLM-L6-v2 ist primär englisch trainiert.
"""
import os
from config import load_config
from rag import RagRetriever

DOCS = [
    "Hate speech based on race dehumanizes people using slurs or promotes racial superiority.",
    "Religious hate speech mocks or calls for violence against people based on faith.",
    "Gender hate speech uses stereotypes to demean women or LGBTQ+ people.",
    "Hate targeting nationality portrays immigrants as criminals or invaders.",
    "Disability hate speech mocks people with physical or mental disabilities.",
    "Antisemitism includes Holocaust denial, conspiracy theories, dehumanizing portrayals.",
    "Hateful memes combine innocent images with text to create hateful meaning through juxtaposition.",
    "Dog whistles are coded language conveying hateful messages with plausible deniability.",
    "Benign statements become hateful combined with images targeting a protected group.",
    "Humor or irony framing does not exempt content from being hate speech.",
    "Dehumanization compares groups to vermin, parasites, or animals.",
    "Historical enemy comparisons imply a group is dangerous or untrustworthy.",
    "Calls for exclusion, segregation, or violence against a protected group are hate speech.",
    "Stereotypes portraying groups as lazy, criminal, greedy, or deviant are hate speech.",
]

if __name__ == "__main__":
    cfg = load_config()
    db_path = os.path.join(cfg.get("results_folder", "."), "chroma_db")
    os.makedirs(db_path, exist_ok=True)
    r = RagRetriever(db_path)
    r.add_documents(DOCS)
    print(f"ChromaDB: {r.count()} Dokumente in {db_path}")
