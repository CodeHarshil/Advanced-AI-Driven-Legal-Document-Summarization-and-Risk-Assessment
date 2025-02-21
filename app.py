import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from datetime import datetime

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

st.header("Advanced AI-Driven Legal Document Summarization and Risk Assessment")

# Add helpful instructions
with st.expander("How to use this chat"):
    st.markdown("""
    1. Upload your legal documents using the sidebar
    2. Click 'Generate Summary' to print summary
    3. Ask questions about your documents in the chat
    4. The AI will maintain context throughout the conversation
    """)

# Initialize session state for summary
if "summary" not in st.session_state:
    st.session_state.summary = None

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
# def input(user_question):
#     response=st.session_state.conversation({"question":user_question})
#     st.write(response)

# setting pre-define questions
# user_question = st.text_input("Ask questions about your Docs")
# if user_question:
#     input(user_question)

def generate_summary(text):
    """Generate summary using LLM"""
    summary_prompt = f"""
    Please provide a comprehensive summary of the following document. Include:
    1. Key Points
    2. Main Findings
    3. Important Dates
    4. Critical Issues
    5. Recommendations

    Document text:
    {text}
    
    Please provide the summary in a clear, well-structured format.
    """
    
    try:
        summary = llm.predict(summary_prompt)
        return summary
    
    except Exception as e:
        st.error(f"Error generating summary")
        return None

def download_summary(summary):
    summary_dict = {"Document Summary": summary}
     
    # Convert the summary_dict to plain text (multi-line formated string)
    summary_str = f"""
    Document Summary:
    {summary_dict['Document Summary']}
    """

    return summary_str
    

with st.sidebar:
    st.subheader("Your Docs")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True)

# Generating Spinners
if st.button("Generate Summary"):
    with st.spinner("Processing documents..."):

        # Get pdf text
        raw_text=getting_text(uploaded_files)

        # get the text chunks
        chunks=getting_chunks(raw_text)

        with st.spinner("Generating summary..."):
            summary = generate_summary(raw_text)
            if summary:
                st.session_state.summary = summary
                st.success("Summary generated successfully!")

# Display and download section
if st.session_state.summary:
    st.header("Document Summary")
    
    # Display summary
    st.markdown(st.session_state.summary)
    
    # Create download button
    formatted_summary = download_summary(st.session_state.summary)    
    st.download_button(
        label="Download Summary",
        data=formatted_summary,
        file_name="document_summary.txt",
        mime="text/plain"
    )

    # Clear summary button
    if st.button("Clear Summary"):
        st.session_state.summary = None
        st.rerun()

else:
    st.info("Upload PDF documents and click 'Generate Summary' to begin.")

