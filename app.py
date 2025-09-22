import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pdfplumber
from docx import Document
import os
import tempfile
import requests
import zipfile

# ==============================
# App Config
# ==============================
st.set_page_config(page_title="Bill & Legal Doc Summarizer", layout="wide")
st.title("Bill & Legal Document Summarizer")
st.markdown("Upload a PDF, DOCX, or TXT file, or paste text to get a **concise summary** of bills, legal documents, or contracts.")

# ==============================
# Load Model from GitHub
# ==============================
@st.cache_resource(show_spinner=True)
def load_model_from_github():
    GITHUB_ZIP_URL = "https://github.com/YourUsername/YourRepoName/archive/refs/heads/main.zip"  # Replace with your GitHub repo zip URL
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "model.zip")

    # Download the zip
    with open(zip_path, "wb") as f:
        f.write(requests.get(GITHUB_ZIP_URL).content)

    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Assuming model is inside a folder like YourRepoName-main/SummifyAI
    model_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0], "SummifyAI")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

with st.spinner("Loading model from GitHub..."):
    tokenizer, model, device = load_model_from_github()
st.success("Model loaded successfully!")

# ==============================
# Helper Functions
# ==============================
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def chunk_text(text, max_words=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks

def summarize_text(text, max_input_len=1024, max_output_len=300):
    chunks = chunk_text(text, max_words=500)
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(chunk, max_length=max_input_len, truncation=True, return_tensors="pt").to(device)
        summary_ids = model.generate(
            **inputs,
            max_length=max_output_len,
            min_length=50,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    final_summary = " ".join(summaries)
    return final_summary

# ==============================
# File Uploader & Input
# ==============================
uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt"])
input_text = ""

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "pdf":
        input_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        input_text = extract_text_from_docx(uploaded_file)
    elif file_type == "txt":
        input_text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type!")

manual_text = st.text_area("Or paste text here:", height=150)
if manual_text:
    input_text += "\n" + manual_text

if st.button("Generate Summary"):
    if not input_text.strip():
        st.warning("Please upload a document or paste text first.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_text(input_text)
        st.success("Summary Generated!")
        st.text_area("Summary Output", summary, height=300)
