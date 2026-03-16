import os
from dotenv import load_dotenv
import streamlit as st
from rag_pipeline import create_vector_store, generate_answer

load_dotenv()

st.title("PDF RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    vectorstore, num_pages, chunks = create_vector_store("temp.pdf")

    st.write("PDF loaded successfully!")
    st.write(f"Number of pages loaded: {num_pages}")

    question = st.text_input("Ask a question about the PDF")

    if question:
        answer = generate_answer(vectorstore, question)

        st.header("AI Answer")
        st.write(answer)

    st.write("Vector database created successfully!")

    st.write(f"Total chunks created: {len(chunks)}")

    st.subheader("Preview of first chunk")
    st.write(chunks[0].page_content)

   
