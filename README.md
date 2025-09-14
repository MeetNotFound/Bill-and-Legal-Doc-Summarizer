# Bill & Legal Document Summarizer

An AI-powered **Bill & Legal Document Summarizer** that automatically generates **concise summaries** of lengthy bills, legislations, and legal documents.  
This project is powered by a **fine-tuned Transformer model (SummifyAI)** on the [BillSum dataset](https://huggingface.co/datasets/billsum), making it highly optimized for legal text summarization.

---

## Features

**Upload Documents** – Supports **PDF, DOCX, TXT**  
**Paste Text** – Copy-paste any legal content directly  
**AI Summarization** – Generates concise, readable summaries  
**Fast & Lightweight** – Runs entirely in the browser using Streamlit  
**Fine-Tuned Model** – Trained specifically on US Congressional Bills for better legal summarization  

---

## Model Details

- **Model Name:** `SummifyAI`  
- **Base Model:** `facebook/bart-large-cnn` (fine-tuned on BillSum)  
- **Training:** 50 Epochs on Kaggle GPU (NVIDIA T4)  
- **Specialization:** Legal/Bill summarization  
- **Inference Device:** Auto (CUDA if available, otherwise CPU)

---

## Installation & Setup

Clone this repository and install dependencies:

```bash
git clone https://github.com/MeetNotFound/Bill-and-Legal-Doc-Summarizer
cd Bill-and-Legal-Doc-Summarizer

pip install -r requirements.txt
```

---

## Run Locally

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

---

## Deployment

This project can be deployed on:

- **Streamlit Cloud** – Free hosting, easy 1-click deployment
- **Hugging Face Spaces** – Free, optimized for ML demos
- **Render / Railway** – Alternative cloud platforms

---

## Project Structure

```
Bill-and-Legal-Doc-Summarizer/
│
├── app.py                 # Streamlit UI & main logic
├── requirements.txt       # Dependencies
├── SummifyAI/             # Fine-tuned model folder
│   ├── config.json
│   ├── tokenizer.json
│   ├── model.safetensors
│   └── ...
└── README.md
```

---

## Example Output

**Input:**  
*"To amend the Internal Revenue Code to allow a refundable tax credit for the purchase of qualified health insurance by individuals not covered under employer-sponsored health plans."*

**Generated Summary:**  
*"Amends the Internal Revenue Code to allow a refundable tax credit for individuals without employer-sponsored health coverage."*

---

## Dataset

This model is fine-tuned on the **[BillSum dataset](https://huggingface.co/datasets/billsum)**:  
> "BillSum is a dataset of US Congressional and California state bills, paired with human-written summaries."

---


## Future Improvements

- Add **OCR support** for scanned legal documents  
- Support **multiple languages**  
- Improve UI with document preview & highlighting  
