import streamlit as st
import base64
import pandas as pd
import torch
import os
from PyPDF2 import PdfReader
from transformers import BertConfig, AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Function to encode background image
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load background image
bg_image = get_base64_image("background_concept19.jpg")

# Page configuration
st.set_page_config(page_title="AskFi", layout="wide")

# ------------------- CSS Styling -------------------
st.markdown(f"""
    <style>
    @font-face {{
        font-family: 'Special Gothic Expanded One';
        src: url('fonts/SpecialGothicExpandedOne-Regular.ttf') format('truetype');
    }}
    html, body, .stApp {{
        background: url("data:image/jpg;base64,{bg_image}") no-repeat center center fixed;
        background-size: cover;
        color: #fff;
        font-family: 'Gunterz', sans-serif;
    }}
    .title-text {{
        font-family: 'Special Gothic Expanded One';
        font-size: 6em;
        text-align: center;
        margin-top: 0.9em;
        color: white;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.8);
    }}
    .dropdown-style {{
        background-color: rgba(255, 192, 203, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 2rem;
    }}
    .answer-box {{
        background: rgba(0, 0, 0, 0.6);
        padding: 1em;
        border-radius: 12px;
        margin-top: 1em;
        font-size: 1.2em;
    }}
    .copyright {{
        position: fixed;
        bottom: 10px;
        right: 15px;
        font-size: 0.8em;
        color: #fff;
        opacity: 0.8;
        z-index: 10;
    }}
    .stButton button {{
        background-color: rgba(255, 255, 255, 0.2);
        color: #000;  
        font-weight: bold;
        border: 2px solid white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-size: 1.1rem;
        transition: background 0.3s ease, color 0.3s ease;
    }}
    .stButton button:hover {{
        background-color: rgba(255, 255, 255, 0.35);
        color: black;
        border-color: #fff;
    }}
    .short-selectbox .stSelectbox {{
        max-width: 300px;
        margin: 0 auto;
    }}
    label, .stSelectbox label, .stTextInput label {{
        color: white !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ------------------- Load Models -------------------
model_dir = r"C:\\Users\\Armaan Raghav\\Desktop\\Internship\\Project_Internship\\bert_banking_model_v2"
config = BertConfig.from_json_file(os.path.join(model_dir, "config.json"))
model = AutoModelForQuestionAnswering.from_pretrained(model_dir, config=config, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)

# Load QA Data
df = pd.read_csv("combined_banking_qa.csv")
df = df.dropna(subset=['question', 'context', 'answer_text'])
qa_data = df.set_index('question')[['context', 'answer_text']].to_dict(orient='index')

# Semantic Search
contexts = df['context'].dropna().unique().tolist()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedder._target_device = torch.device("cpu")
context_embeddings = embedder.encode(contexts, convert_to_numpy=True)
nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(context_embeddings)

def get_best_context(question):
    q_embed = embedder.encode([question])[0]
    dist, idx = nn.kneighbors([q_embed])
    return contexts[idx[0][0]]

# ------------------- UI -------------------
st.markdown("<div class='title-text'></div>", unsafe_allow_html=True)

# Section selector
with st.container():
    st.markdown('<div class="short-selectbox">', unsafe_allow_html=True)
    tab = st.selectbox("Choose a section:", ["FAQs", "PDF Summary"], index=0, key="menu", help="Select an option to explore features.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- FAQs Section -------------------
if tab == "FAQs":
    selected_faq = st.selectbox("Suggested FAQs:", ["--Select a question--"] + list(qa_data.keys()), key="faq_selectbox")
    user_question = st.text_input("Ask Your Question:")
    final_question = user_question.strip() if user_question.strip() else (selected_faq if selected_faq != "--Select a question--" else "")

    if st.button("Get Answer", key="get_answer"):
        if final_question == "":
            st.error("Please type or select a question.")
        elif final_question in qa_data:
            answer = qa_data[final_question]['answer_text']
        else:
            context = get_best_context(final_question)
            result = qa_pipeline(question=final_question, context=context)
            answer = result['answer']

        st.markdown(f"<div class='answer-box'><b>Answer:</b><br>{answer}</div>", unsafe_allow_html=True)

# ------------------- PDF Summary Section -------------------
elif tab == "PDF Summary":
    st.subheader("Upload your PDF to generate a summary")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.button("Get Summary", key="get_summary"):
        if uploaded_pdf is not None:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            with st.spinner("Summarizing your PDF..."):
                reader = PdfReader(uploaded_pdf)
                text = "".join(page.extract_text() or "" for page in reader.pages)
                chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
                summary = " ".join(summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text'] for chunk in chunks)
            st.markdown(f"<div class='answer-box'><b>PDF Summary:</b><br>{summary.strip()}</div>", unsafe_allow_html=True)
        else:
            st.error("Please upload a PDF to summarize.")

# ------------------- Footer -------------------
st.markdown("<div class='copyright'>Project made by - Arman Raghav</div>", unsafe_allow_html=True)
