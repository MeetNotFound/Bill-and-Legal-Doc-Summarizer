import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pdfplumber
from docx import Document

# ---------------------------
# Load your fine-tuned model
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("SummifyAI")
    tokenizer = AutoTokenizer.from_pretrained("SummifyAI")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

# ---------------------------
# Helper functions
# ---------------------------
def summarize_chunks(text, max_input_len=1024, max_output_len=130):
    """Summarize a single chunk of text"""
    inputs = tokenizer(text, max_length=max_input_len, truncation=True, return_tensors="pt").to(device)
    summary_ids = model.generate(
        **inputs,
        max_length=max_output_len,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_pages(pages_text, max_input_len=1024, max_output_len=130):
    """Summarize each page separately, then combine summaries"""
    page_summaries = []

    for page_text in pages_text:
        # Split into sentence-based chunks to fit the model
        sentences = page_text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            token_length = len(tokenizer(current_chunk + sentence, return_tensors="pt")["input_ids"][0])
            if token_length < max_input_len:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Summarize each chunk of the page
        chunk_summaries = [summarize_chunks(chunk, max_input_len, max_output_len) for chunk in chunks]
        page_summaries.append(" ".join(chunk_summaries))

    # Combine all page summaries
    combined_summary = " ".join(page_summaries)

    # Optional: Final summarization for conciseness
    if len(tokenizer(combined_summary, return_tensors="pt")["input_ids"][0]) > max_input_len:
        combined_summary = summarize_chunks(combined_summary, max_input_len, max_output_len)

    return combined_summary

def extract_text_from_pdf(file):
    """Extract text from PDF and return list of page texts"""
    pages_text = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return pages_text

def extract_text_from_docx(file):
    """Extract text from DOCX and split by paragraphs as 'pages'"""
    pages_text = []
    try:
        doc = Document(file)
        para_text = "\n".join([para.text for para in doc.paragraphs])
        # Split DOCX into pseudo-pages every 500 words
        words = para_text.split()
        for i in range(0, len(words), 500):
            pages_text.append(" ".join(words[i:i + 500]))
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
    return pages_text

def extract_text_from_txt(file):
    """Extract text from TXT file and split into pseudo-pages"""
    text = file.read().decode("utf-8")
    words = text.split()
    pages_text = []
    for i in range(0, len(words), 500):
        pages_text.append(" ".join(words[i:i + 500]))
    return pages_text

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Bill & Legal Doc Summarizer", layout="wide")
st.title("Bill & Legal Document Summarizer")
st.markdown("Upload a PDF, DOCX, TXT file, or paste text to get a **concise summary** of bills, legal documents, or contracts.")

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt"])

pages_text = []

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "pdf":
        pages_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        pages_text = extract_text_from_docx(uploaded_file)
    elif file_type == "txt":
        pages_text = extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file type!")

manual_text = st.text_area("Or paste text here:", height=150)
if manual_text:
    # Treat manual input as a single page
    pages_text.append(manual_text)

if st.button("Generate Summary"):
    if not pages_text:
        st.warning("Please upload a document or paste text first.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_pages(pages_text)
        st.success("Summary Generated!")
        st.subheader("Summary Output")
        st.write(summary)
