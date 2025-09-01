import streamlit as st
from pdf_utils import extract_text_chunks
from embed_utils import build_faiss_index, search
from qa_engine import generate_answer

st.title("📚 StudyMate – AI PDF Q&A")

uploaded_file = st.file_uploader("Upload your study material (PDF)", type="pdf")
question = st.text_input("Ask a question about your material")

if uploaded_file and question:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    chunks = extract_text_chunks("temp.pdf")

    if not chunks:
        st.error("No text could be extracted from the PDF. Please upload a valid document.")
        st.stop()

    index, embeddings, chunk_list = build_faiss_index(chunks)
    relevant_chunks = search(question, index, chunk_list, embeddings)
    answer = generate_answer(relevant_chunks, question)

    st.subheader("Answer")
    st.write(answer)
