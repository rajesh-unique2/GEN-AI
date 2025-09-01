import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_faiss_index(chunks):
    if not chunks or not isinstance(chunks, list):
        raise ValueError("Chunk list is empty or invalid. Cannot build FAISS index.")

    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings)

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    elif embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings, chunks

def search(query, index, chunks, embeddings, top_k=3):
    query_vec = model.encode([query], show_progress_bar=False)
    query_vec = np.array(query_vec)

    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]
