# 📁 File name: extract_pdf_text.py
# ✅ Purpose: Extracts text from your banking PDF documents

import fitz  # PyMuPDF
import os

# Function to extract text from a single PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract and save text from multiple PDFs
def extract_all_pdfs(folder_path, output_file="extracted_texts.txt"):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    
    with open(output_file, "w", encoding="utf-8") as out:
        for pdf in pdf_files:
            full_path = os.path.join(folder_path, pdf)
            print(f"Extracting from: {pdf}")
            text = extract_text_from_pdf(full_path)
            out.write(f"\n--- START OF {pdf} ---\n\n")
            out.write(text)
            out.write(f"\n--- END OF {pdf} ---\n\n")
    print(f"\n✅ All PDFs extracted and saved to {output_file}")

# 🔁 Example usage
if __name__ == "__main__":
    # Replace this path with the folder where your PDFs are stored
    pdf_folder = "./pdfs"  # Example: place your PDFs inside a folder named 'pdfs' next to this script
    extract_all_pdfs(pdf_folder)
