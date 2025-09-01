import fitz  # PyMuPDF

def extract_text_chunks(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        text = page.get_text()
        if text:
            full_text += text + "\n"

    doc.close()

    # If no text was extracted, return empty list
    if not full_text.strip():
        return []

    # Split into chunks
    chunks = []
    words = full_text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
