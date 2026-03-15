import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

st.title("PDF RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    st.write("PDF loaded successfully!")
    st.write(f"Number of pages loaded: {len(documents)}")

    st.subheader("First page preview:")
    st.write(documents[0].page_content[:1000])
