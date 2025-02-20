import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

st.header("Advanced AI-Driven Legal Document Summarization and Risk Assessment")

# Getting text of pdfs
def getting_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)   # Reading from pdf
        for page in pdf_reader.pages:   #taking texts from pdf's pages
            text = text + page.extract_text()
    return text

def getting_chunks(text):
    text_splitters = CharacterTextSplitter(separator="\n",chunk_size=500,chunk_overlap=100,length_function=len)
    chunks=text_splitters.split_text(text)
    return chunks

# Handeling user input
def input(user_question):
    response=st.session_state.conversation({"question":user_question})
    st.write(response)

# setting pre-define questions
user_question = st.text_input("Ask questions about your Docs")
if user_question:
    input(user_question)


# Add helpful instructions
with st.expander("How to use this chat"):
    st.markdown("""
    1. Upload your legal documents using the sidebar
    2. Click 'Create Chunks' to visualize chunks
    3. Ask questions about your documents in the chat
    4. The AI will maintain context throughout the conversation
    """)

with st.sidebar:
    st.subheader("Your Docs")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True)
    if st.button("Create Chunks"):
        with st.spinner("Creating chunks..."):

            # Get pdf text
            raw_text=getting_text(uploaded_files)

            # get the text chunks
            chunks=getting_chunks(raw_text)
            st.write(chunks)


