# AskFi – Banking FAQ & PDF Assistant  

![AskFi Banner](assets/gifs/banner.gif)

---

## Overview  

**AskFi** is an AI-powered banking assistant that helps users:  

- Ask banking-related questions and get accurate answers  
- Summarize PDFs such as FAQs, loan agreements, and financial documents  
- Interact via a sleek Streamlit interface with a modern, user-friendly design  

Powered by **BERT, FAISS, and SentenceTransformers**, AskFi is optimized for semantic search, Q&A, and summarization.  

---

## Technologies Used  

- Fine-tuned BERT – Question Answering  
- FAISS – Semantic Search Index  
- Streamlit – Interactive Web App  
- SentenceTransformers – Embedding Generation  
- PyPDF2 – PDF Processing  

---

## Features  

- **Banking QA Chatbot** – Context-aware answers to FAQs  
- **PDF Summarization** – Upload and summarize large documents  
- **Semantic Search** – Retrieve the most relevant sections quickly  
- **Clean UI** – Dropdown navigation (FAQs / PDF Summary)  
- **Curated Dataset** – High-quality Q&A pairs for fine-tuning  

![Feature Demo](assets/gifs/demo.gif)

---

## Project Structure  

```markdown
AskFi/
├── app.py # Streamlit main application
├── faiss_index.index # FAISS index for semantic search
├── chunked_texts.pkl # Chunked PDF text embeddings
├── requirements.txt # Python dependencies
├── assets/
│ ├── background_concept5.jpg
│ ├── background_concept4.jpg
│ └── gifs/
│ ├── banner.gif
│ └── demo.gif
├── models/
│ └── fine_tuned_bert/ # Fine-tuned QA model
├── data/
│ ├── banking_faq.csv
│ └── extracted_texts.txt
└── README.md
```


---

## Installation  

1. **Clone the repository**  
```
git clone https://github.com/yourusername/AskFi.git
```
```
cd AskFi
```
2. **Create a virtual environment**  
```
python -m venv venv
```
3. **Activate the environment**  

- **Windows**  
  ```
  venv\Scripts\activate
  ```

- **Linux / Mac**  
  ```
  source venv/bin/activate
  ```

4. **Install dependencies**
```  
pip install -r requirements.txt
```

---

## Usage  

Run the Streamlit application:  
```
streamlit run app.py
```

- Select **FAQs** to ask banking questions  
- Select **PDF Summary** to upload and summarize a document  
- Enter your query and click **Generate**  
- For PDFs, upload the file and click **Get Summary**  

---

## How It Works  

### Banking QA  
1. User asks a question  
2. Question embeddings are generated with SentenceTransformer  
3. FAISS retrieves the best context from `chunked_texts.pkl`  
4. The fine-tuned BERT model generates an answer  

### PDF Summarization  
1. User uploads a PDF  
2. Text is extracted and chunked  
3. Embeddings created and indexed with FAISS  
4. Summarization model generates a concise summary  

---

## Dataset  

- Custom banking FAQ dataset with Q&A triplets  
- Supports embedding-based search and PDF summarization  

---

## Dependencies  

- streamlit  
- transformers  
- sentence-transformers  
- faiss-cpu  
- PyPDF2  
- torch  
- pandas  
- numpy  

All dependencies are listed in `requirements.txt`.  

---

## Commands Summary  

| Task                        | Command                                |
|-----------------------------|----------------------------------------|
| Run Streamlit app           | `streamlit run app.py`                 |
| Install dependencies        | `pip install -r requirements.txt`      |
| Activate venv (Windows)     | `venv\Scripts\activate`                |
| Activate venv (Linux/Mac)   | `source venv/bin/activate`             |

---

## Future Enhancements  

- Multi-language support  
- Real-time banking news integration  
- Advanced GPT-based summarization  
- User authentication for secure PDF access  

---

## License  

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.  

---

## Contact  

**Arman Raghav**  
Email: [armanraghavwork@gmail.com]()  
GitHub: [https://github.com/yourusername](https://github.com/yourusername)  

---
