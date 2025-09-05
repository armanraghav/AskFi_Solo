import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the text
with open("extracted_texts.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# Simple chunking logic
def chunk_text(text, max_len=300, overlap=50):
    sentences = text.split('. ')
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_len:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

chunks = chunk_text(full_text)

# Save chunks
with open("chunked_texts.pkl", "wb") as f:
    pickle.dump(chunks, f)

# Embed chunks
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "faiss_index.bin")
print("✅ FAISS index and chunks saved.")
