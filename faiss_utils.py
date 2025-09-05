import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index and embeddings
def load_faiss_index(index_path, mapping_path, model_name="all-MiniLM-L6-v2"):
    index = faiss.read_index(index_path)
    with open(mapping_path, 'rb') as f:
        chunks = pickle.load(f)
    embedder = SentenceTransformer(model_name)
    return index, chunks, embedder

# Search the index for most similar chunk
def search_faiss_index(question, index, chunks, embedder, top_k=1):
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(np.array(question_embedding), top_k)
    return [chunks[i] for i in indices[0]]
