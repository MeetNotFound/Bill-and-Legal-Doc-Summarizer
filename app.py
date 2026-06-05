import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pdfplumber
from docx import Document
from PIL import Image
import easyocr
import numpy as np
import tempfile
import os
import PyPDF2

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="SummifyAI OCR",
    page_icon="📄",
    layout="wide"
)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.title("📄 SummifyAI OCR")

    st.markdown("""
### Supported Files

- PDF
- DOCX
- TXT
- PNG
- JPG
- JPEG

### Features

- Fine-Tuned Transformer
- OCR Extraction
- Legal Document Summarization
- Bill Summarization
- Long Document Chunking
- Downloadable Summaries
""")

# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource
def load_model():

    MODEL_HF_REPO = "MeetNotFound/Bill-and-Legal-Doc-Summarizer"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_REPO)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_HF_REPO
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    return model, tokenizer, device


with st.spinner("Loading SummifyAI model..."):
    model, tokenizer, device = load_model()

# =====================================================
# OCR
# =====================================================

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

# =====================================================
# TEXT EXTRACTION
# =====================================================

def extract_text_from_pdf(uploaded_pdf):

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    temp_file.write(uploaded_pdf.read())
    temp_file.flush()

    pdf_reader = PyPDF2.PdfReader(temp_file.name)

    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    temp_file.close()
    os.remove(temp_file.name)

    return text.strip()


def extract_text_from_docx(uploaded_docx):

    doc = Document(uploaded_docx)

    return "\n".join(
        [para.text for para in doc.paragraphs]
    )


def extract_text_from_txt(uploaded_txt):

    return uploaded_txt.read().decode("utf-8")


def extract_text_from_image(uploaded_image):

    image = Image.open(uploaded_image).convert("RGB")

    image_np = np.array(image)

    result = reader.readtext(
        image_np,
        detail=0
    )

    return "\n".join(result)

# =====================================================
# CHUNKING
# =====================================================

def chunk_text(text, max_words=500):

    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        max_words
    ):
        chunks.append(
            " ".join(
                words[i:i + max_words]
            )
        )

    return chunks

# =====================================================
# SUMMARIZATION
# =====================================================

def summarize_text(text):

    chunks = chunk_text(text)

    summaries = []

    for chunk in chunks:

        inputs = tokenizer(
            chunk,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        summary_ids = model.generate(
            **inputs,
            max_length=150,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

        summaries.append(
            tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )
        )

    return " ".join(summaries)

# =====================================================
# HEADER
# =====================================================

st.title("📄 SummifyAI OCR")

st.caption(
    "AI-powered Legal, Bill, Contract and Document Summarization using a Fine-Tuned Transformer Model"
)

# =====================================================
# UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload a Document or Image",
    type=[
        "pdf",
        "docx",
        "txt",
        "png",
        "jpg",
        "jpeg"
    ]
)

# =====================================================
# PROCESS
# =====================================================

if uploaded_file is not None:

    with st.spinner("Extracting content..."):

        file_type = uploaded_file.name.split(".")[-1].lower()

        if file_type == "pdf":

            extracted_text = extract_text_from_pdf(
                uploaded_file
            )

            st.success("PDF processed successfully")

        elif file_type == "docx":

            extracted_text = extract_text_from_docx(
                uploaded_file
            )

            st.success("DOCX processed successfully")

        elif file_type == "txt":

            extracted_text = extract_text_from_txt(
                uploaded_file
            )

            st.success("TXT processed successfully")

        else:

            extracted_text = extract_text_from_image(
                uploaded_file
            )

            st.success(
                "OCR extraction completed"
            )

    if extracted_text:

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Characters",
                len(extracted_text)
            )

        with col2:
            st.metric(
                "Words",
                len(extracted_text.split())
            )

        tab1, tab2 = st.tabs(
            [
                "Extracted Text",
                "Summary"
            ]
        )

        with tab1:

            st.text_area(
                "Extracted Text",
                extracted_text,
                height=350
            )

        with tab2:

            if st.button(
                "Generate Summary"
            ):

                with st.spinner(
                    "Generating summary..."
                ):

                    summary = summarize_text(
                        extracted_text
                    )

                st.success(
                    "Summary generated successfully"
                )

                st.text_area(
                    "Summary",
                    summary,
                    height=250
                )

                st.download_button(
                    "Download Summary",
                    summary,
                    file_name="summary.txt"
                )
