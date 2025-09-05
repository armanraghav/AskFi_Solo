from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Load chunked texts
with open("chunked_texts.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

# Step 2: Load sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Encode chunks into embeddings
embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

# Step 4: Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 5: Save FAISS index and metadata
faiss.write_index(index, "faiss_index.bin")

with open("faiss_chunks.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n")

print("✅ FAISS index and chunks saved successfully.")
