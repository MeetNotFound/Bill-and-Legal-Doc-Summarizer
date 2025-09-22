import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pdfplumber
from docx import Document

# ==============================
# App Config
# ==============================
st.set_page_config(page_title="Summify AI", layout="wide")
st.title("Summify AI - Document Summarizer")
st.markdown("Upload a PDF, DOCX, or TXT file, or paste text to get a **concise summary** of bills, legal documents, or contracts.")

# ==============================
# Load Model from GitHub
# ==============================
@st.cache_resource(show_spinner=True)
def load_model():
    MODEL_HF_REPO = "MeetNotFound/Bill-and-Legal-Doc-Summarizer"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_REPO)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_HF_REPO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

with st.spinner("Loading model from GitHub..."):
    tokenizer, model, device = load_model()
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
    """Split text into chunks of ~500 words"""
    words = text.split()
    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks

def summarize_text(text, max_input_len=1024, max_output_len=300):
    """Summarize text using chunking to avoid truncation"""
    chunks = chunk_text(text, max_words=500)
    chunk_summaries = []

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            max_length=max_input_len,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        summary_ids = model.generate(
            **inputs,
            max_length=max_output_len,
            min_length=50,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        chunk_summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    # Combine chunk summaries
    final_summary = " ".join(chunk_summaries)

    # Re-summarize if still too long
    token_length = len(tokenizer(final_summary, return_tensors="pt")["input_ids"][0])
    if token_length > max_input_len:
        inputs = tokenizer(
            final_summary,
            max_length=max_input_len,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        summary_ids = model.generate(
            **inputs,
            max_length=max_output_len,
            min_length=50,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return final_summary

# ==============================
# Streamlit UI - File Upload
# ==============================
uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
input_text = ""

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "pdf":
        st.write("**PDF detected — extracting text directly**")
        input_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        st.write("**DOCX detected — extracting text**")
        input_text = extract_text_from_docx(uploaded_file)
    elif file_type == "txt":
        st.write("**TXT file detected**")
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
        st.subheader("Summary Output")
        st.text_area("Summary", summary, height=300)
