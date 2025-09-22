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

def summarize_pages(pages_text, max_input_len=1024, max_output_len=130):
    """Summarize each page separately, then combine summaries"""
    page_summaries = []

    for page_text in pages_text:
        # Split into smaller sentence-based chunks for the page
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

        # Combine chunk summaries for the page
        page_summary = " ".join(chunk_summaries)
        page_summaries.append(page_summary)

    # Combine all page summaries into one text
    combined_summary = " ".join(page_summaries)

    # Optional: Final summary for conciseness
    if len(tokenizer(combined_summary, return_tensors="pt")["input_ids"][0]) > max_input_len:
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
