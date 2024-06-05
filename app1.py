import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

from dotenv import load_dotenv
load_dotenv()

# Load the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

def vector_embedding():
    if "vectors" not in st.session_state:
        try:
            st.session_state.embeddings = NVIDIAEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
            st.session_state.docs = st.session_state.loader.load()  # Document Loading
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # Splitting
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
        except Exception as e:
            st.error(f"Error during vector embedding: {e}")

st.title("Extract Documents Using Nvidia NIM")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
""")

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

if prompt1 and "vectors" in st.session_state:
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start} seconds")
        st.write(response['answer'])

        # With a Streamlit expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    except Exception as e:
        st.error(f"Error during retrieval: {e}")
else:
    if prompt1:
        st.error("Please create the vector embeddings first by clicking the 'Documents Embedding' button.")
