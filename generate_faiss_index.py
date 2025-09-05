import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os

# Load your chunked texts
with open("chunked_texts.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_numpy=True)

# Save the chunks for use in the web app
with open("chunked_texts.pkl", "wb") as f:
    pickle.dump(chunks, f)

# Create and save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "faiss_index.index")

print("FAISS index and chunked texts saved successfully.")
