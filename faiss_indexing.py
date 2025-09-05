import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load chunked texts
with open("chunked_texts.txt", "r", encoding="utf-8") as file:
    chunks = [line.strip() for line in file if line.strip()]

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all chunks to vectors
print("Generating embeddings...")
embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

# Normalize embeddings (recommended for cosine similarity)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine sim after normalization
index.add(embeddings)

# Save index
faiss.write_index(index, "faiss_index.bin")

# Save corresponding chunks to match later
with open("chunk_texts.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ FAISS index created and saved as 'faiss_index.bin'")
