import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------
# Global embedding model (loaded once at import)
# -----------------------------------------------------
print("[INFO] Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[INFO] Embedding model loaded successfully.\n")


# -----------------------------------------------------
# Core functions
# -----------------------------------------------------
def build_faiss_index(dataset_path: str, index_path: str):
    """Build FAISS index from dataset JSON and save it"""
    print(f"[INFO] Building FAISS index for dataset: {dataset_path}")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Combine question + answer if present, else use 'text'
    docs = []
    for entry in dataset:
        if "text" in entry:
            docs.append(entry["text"])
        else:
            q = entry.get("question", "")
            a = entry.get("answer", "")
            docs.append(f"{q} {a}".strip())

    print(f"[INFO] Number of documents to index: {len(docs)}")

    embeddings = _embedding_model.encode(docs, convert_to_numpy=True)
    print("[INFO] Embeddings computed.")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print(f"[INFO] FAISS index created with dimension: {embeddings.shape[1]}")

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"[INFO] FAISS index saved at: {index_path}\n")

    return index, docs


def load_or_build_index(usecase: str):
    """Ensure FAISS index exists; build if missing"""
    dataset_path = f"datasets/knowledge_base_cache/{usecase}.json"
    index_path = f"datasets/indexes/{usecase}.index"

    if os.path.exists(index_path):
        print(f"[INFO] Loading existing FAISS index: {index_path}")
        index = faiss.read_index(index_path)
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        docs = []
        for entry in dataset:
            if "text" in entry:
                docs.append(entry["text"])
            else:
                q = entry.get("question", "")
                a = entry.get("answer", "")
                docs.append(f"{q} {a}".strip())

        print(f"[INFO] Loaded {len(docs)} documents from dataset.\n")
        return index, docs

    elif os.path.exists(dataset_path):
        print(f"[INFO] No index found. Building new FAISS index for usecase '{usecase}'...")
        return build_faiss_index(dataset_path, index_path)

    else:
        raise FileNotFoundError(
            f"Neither index nor dataset found for usecase '{usecase}'"
        )


def rag_retrieve(query: str, usecase: str, top_k: int = 3, embedder=None, index=None, docs=None):
    """Retrieve top-k relevant docs for query from a given index (if provided)."""
    print(f"[INFO] Retrieving RAG context for query: '{query}' (usecase: {usecase}, top_k={top_k})")

    # Use preloaded index and docs if available
    if index is None or docs is None:
        index, docs = load_or_build_index(usecase)

    # Use global embedder if not provided
    if embedder is None:
        embedder = get_embedder()

    query_embedding = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)
    retrieved = [docs[i] for i in I[0] if i < len(docs)]

    print(f"[INFO] Retrieved {len(retrieved)} documents.")
    return "\n".join(retrieved)



# -----------------------------------------------------
# Helper functions for faster initialization in main_attack.py
# -----------------------------------------------------
def load_index_and_docs(usecase: str):
    """Load FAISS index and documents once for a given usecase"""
    index, docs = load_or_build_index(usecase)
    print(f"[INFO] FAISS index and {len(docs)} documents loaded for usecase '{usecase}'.")
    return index, docs


def get_embedder():
    """Return the global embedding model"""
    return _embedding_model
