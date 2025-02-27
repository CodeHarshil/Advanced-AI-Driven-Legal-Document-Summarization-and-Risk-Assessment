import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

st.header("Advanced AI-Driven Legal Document Summarization and Risk Assessment")

# Add helpful instructions
with st.expander("How to use this chat"):
    st.markdown("""
    1. Upload your legal documents using the sidebar
    2. Click 'Generate Summary' to print summary
    3. Click 'Generate Risk Assessment' to analyze risks
    4. Ask questions about your documents in the chat
    5. The AI will maintain context throughout the conversation
    """)

# Initialize session state variables
if "summary" not in st.session_state:
    st.session_state.summary = None
if "risk_assessment" not in st.session_state:
    st.session_state.risk_assessment = None
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "risk_percentage" not in st.session_state:
    st.session_state.risk_percentage = None

def process_pdfs(uploaded_files):
    """Process PDFs to extract text and create vector store"""
    raw_text = getting_text(uploaded_files)
    chunks = getting_chunks(raw_text)
    vectorstore = get_vectorstore(chunks)
    st.session_state.vectorstore = vectorstore

def getting_text(pdf_docs):
    """Extract text from PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Reading from pdf
        for page in pdf_reader.pages:  #Taking texts from pdf's pages
            text += page.extract_text()
    return text

def getting_chunks(text):
    """Split text into chunks"""
    text_splitters = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100, length_function=len)
    chunks = text_splitters.split_text(text)
    return chunks

def get_vectorstore(chunks):
    """Create and return a vector store from document chunks"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create a conversation chain with memory and retrieval from vector store"""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def generate_summary(vectorstore):
    """Generate summary using LLM and Vector Store"""
    summary_prompt = """
    You are a legal document analyzer. Based on the provided document context, please provide a comprehensive summary of the document. Include:
    1. Key Points
    2. Main Findings
    3. Important Dates
    4. Critical Issues
    5. Recommendations

    Please provide the summary in a clear, well-structured format with headings and bullet points where appropriate.
    """
    try:
        relevant_docs = vectorstore.similarity_search(summary_prompt, k=5)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        final_prompt = summary_prompt + "\n\nDocument context: " + context
        summary = llm.predict(final_prompt)
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def generate_risk_assessment(vectorstore):
    """Generate risk assessment using LLM and keyword checking"""
    risk_prompt = """
    You are a legal risk assessment expert. Based on the provided document context, please identify and analyze potential legal risks in the document. Include:

    1. High-Risk Areas: Identify clauses, terms, or sections that present significant legal exposure
    2. Compliance Issues: Note any potential regulatory or compliance concerns
    3. Liability Analysis: Analyze potential liability arising from the document
    4. Contractual Weaknesses: Identify vague, ambiguous, or problematic contract terms
    5. Recommendations: Suggest specific actions to mitigate identified risks

    Present your analysis in a structured format with clear risk ratings (High/Medium/Low) for each identified issue.
    """
    risk_keywords = ["risk", "liability", "compliance", "exposure", "legal issue"]
    try:
        relevant_docs = vectorstore.similarity_search(risk_prompt, k=5)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        risk_context = ""

        keyword_count = sum(1 for keyword in risk_keywords if keyword in context.lower()) #Counts how many time risk keywords are present
        total_keywords = len(risk_keywords)
        risk_percentage = (keyword_count / total_keywords) * 100
        st.session_state.risk_percentage = risk_percentage
        final_prompt = risk_prompt + "\n\nDocument context: " + context + "\n\nRisk keywords found: " + risk_context
        risk_assessment = llm.predict(final_prompt)
        return risk_assessment
    except Exception as e:
        st.error(f"Error generating risk assessment: {e}")
        return None

def download_summary_pdf(summary):
    """Create a PDF summary with proper text wrapping and pagination"""
    buffer = BytesIO()  #Creates in-memory binary stream which is used to generate pdf
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Document Summary", styles["Title"]),
        Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["BodyText"]),
        Spacer(1, 12)  # Adds 12 points of vertical space
    ]
    story.extend(Paragraph(line, styles["BodyText"]) for line in summary.split("\n")) #Split strings into lines and extends it
    doc.build(story)
    buffer.seek(0)
    return buffer

def download_risk_assessment_pdf(risk_assessment):
    """Create a PDF risk assessment with proper text wrapping and pagination"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Risk Assessment Report", styles["Title"]),
        Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["BodyText"]),
        Spacer(1, 12)
    ]
    story.extend(Paragraph(line, styles["BodyText"]) for line in risk_assessment.split("\n"))
    doc.build(story)
    buffer.seek(0)
    return buffer

def handle_user_question(user_question):
    """Process user question and get response from the conversation chain"""
    response = st.session_state.conversation({"question": user_question, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((user_question, response["answer"]))
    return response["answer"]

# Sidebar for document upload
with st.sidebar:
    st.subheader("Your Docs")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True)

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Summary", "Risk Assessment", "Chat"])

with tab1:
    st.header("Document Summary")
    if st.button("Generate Summary"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                process_pdfs(uploaded_files)
            with st.spinner("Generating summary..."):
                summary = generate_summary(st.session_state.vectorstore)
                if summary:
                    st.session_state.summary = summary
                    st.success("Summary generated successfully!")
        else:
            st.error("Please upload PDF files first.")
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
        pdf_summary = download_summary_pdf(st.session_state.summary)
        st.download_button(
            label="Download Summary as PDF",
            data=pdf_summary,
            file_name="document_summary.pdf",
            mime="application/pdf"
        )
        if st.button("Clear Summary"):
            st.session_state.summary = None
            st.rerun()
    else:
        st.info("Upload PDF documents and click 'Generate Summary' to begin.")

with tab2:
    st.header("Risk Assessment")
    if st.button("Generate Risk Assessment"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                if st.session_state.vectorstore is None:
                    process_pdfs(uploaded_files)
            with st.spinner("Generating risk assessment..."):
                risk_assessment = generate_risk_assessment(st.session_state.vectorstore)
                if risk_assessment:
                    st.session_state.risk_assessment = risk_assessment
                    st.success("Risk assessment generated successfully!")
        else:
            st.error("Please upload PDF files first.")
    if st.session_state.risk_assessment:
        st.markdown(st.session_state.risk_assessment)
        st.write(f"Risk Percentage: {st.session_state.risk_percentage:.2f}%")
        pdf_risk = download_risk_assessment_pdf(st.session_state.risk_assessment)
        st.download_button(
            label="Download Risk Assessment as PDF",
            data=pdf_risk,
            file_name="risk_assessment.pdf",
            mime="application/pdf"
        )
        if st.button("Clear Risk Assessment"):
            st.session_state.risk_assessment = None
            st.session_state.risk_percentage = None
            st.rerun()
    else:
        st.info("Upload PDF documents and click 'Generate Risk Assessment' to begin.")

with tab3:
    st.header("Chat with Your Documents")
    if st.button("Initialize Chat"):
        if uploaded_files:
            with st.spinner("Setting up chat..."):
                if st.session_state.vectorstore is None:
                    process_pdfs(uploaded_files)
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
                st.session_state.processed = True
                st.success("Chat initialized! You can now ask questions about your documents.")
        else:
            st.error("Please upload PDF files first.")
    if st.session_state.processed and st.session_state.conversation:
        for question, answer in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
        user_question = st.chat_input("Ask a question about your documents")
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = handle_user_question(user_question)
                    st.write(response)
            st.rerun()
        if st.button("Clear Chat History") and st.session_state.chat_history:
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Upload PDF documents and click 'Initialize Chat' to start chatting with your documents.")
    