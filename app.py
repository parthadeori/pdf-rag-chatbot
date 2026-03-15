import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

st.title("PDF RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    st.write("PDF loaded successfully!")
    st.write(f"Number of pages loaded: {len(documents)}")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
        )

    # Store embeddings in vector database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    question = st.text_input("Ask a question about the PDF")

    if question:
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(question)

        context = docs[0].page_content

        prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        st.header("AI Answer")
        st.write(response.content)

    st.write("Vector database created successfully!")

    st.write(f"Total chunks created: {len(chunks)}")

    st.subheader("Preview of first chunk")
    st.write(chunks[0].page_content)

   
