import os

# Parameters
input_file = "extracted_texts.txt"
output_file = "chunked_texts.txt"
chunk_size = 300  # number of words per chunk
overlap = 50      # overlapping words between chunks

# Load the extracted text
with open(input_file, "r", encoding="utf-8") as f:
    full_text = f.read()

# Split text into words
words = full_text.split()
chunks = []

# Create overlapping chunks
for i in range(0, len(words), chunk_size - overlap):
    chunk = words[i:i + chunk_size]
    chunk_text = " ".join(chunk).strip()
    if len(chunk_text) > 0:
        chunks.append(chunk_text)

# Save chunks to output file
with open(output_file, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n\n---CHUNK---\n\n")

print(f"✅ Chunking complete! Total chunks created: {len(chunks)}")
print(f"📄 Output saved to: {output_file}")
