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

def chunk_text(text, tokenizer, max_tokens=1024, safety_margin=50):
    """Dynamically split text into chunks based on model's token limit."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        test_chunk = " ".join(current_chunk + [word])
        token_count = len(tokenizer(test_chunk, return_tensors="pt")["input_ids"][0])
        if token_count < (max_tokens - safety_margin):
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_text(text, detail_level='Medium', max_input_len=1024):
    """Summarize long text with adaptive chunking and recursive summarization."""
    # Determine max_output_len based on detail level
    detail_map = {'Short': 150, 'Medium': 300, 'Detailed': 512}
    max_output_len = detail_map.get(detail_level, 300)

    # Chunk text
    chunks = chunk_text(text, tokenizer, max_tokens=max_input_len, safety_margin=50)

    # Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, max_length=max_input_len, truncation=True, return_tensors="pt").to(device)
        summary_ids = model.generate(
            **inputs,
            max_length=max_output_len,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        chunk_summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    combined_summary = " ".join(chunk_summaries)

    # Recursive summarization if still too long
    while len(tokenizer(combined_summary, return_tensors="pt")["input_ids"][0]) > max_input_len:
        inputs = tokenizer(combined_summary, max_length=max_input_len, truncation=True, return_tensors="pt").to(device)
        summary_ids = model.generate(
            **inputs,
            max_length=max_output_len,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        combined_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return combined_summary

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

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Bill & Legal Doc Summarizer", layout="wide")
st.title("Bill & Legal Document Summarizer")
st.markdown(
    "Upload a PDF, DOCX, TXT file, or paste text to get a **concise summary** of bills, legal documents, or contracts."
)

# File upload
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

# Manual text input
manual_text = st.text_area("Or paste text here:", height=150)
if manual_text:
    input_text += "\n" + manual_text

# Detail level slider
detail_level = st.select_slider(
    "Select Summary Detail Level:",
    options=["Short", "Medium", "Detailed"],
    value="Medium"
)

# Generate summary
if st.button("Generate Summary"):
    if not input_text.strip():
        st.warning("Please upload a document or paste text first.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_text(input_text, detail_level=detail_level)
        st.success("Summary Generated!")
        st.subheader("Summary Output")
        st.write(summary)
