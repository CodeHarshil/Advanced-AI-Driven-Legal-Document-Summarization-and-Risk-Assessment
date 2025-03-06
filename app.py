import os
import time
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
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import requests
import json
import base64
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import pickle

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def get_gmail_service():
    """Create and return Gmail API service"""
    creds = None
    
    # Check if token.pickle exists
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If credentials don't exist or are invalid, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=8501)
        
        # Save credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('gmail', 'v1', credentials=creds)

def send_email(recipient_email, subject, body, attachment=None, attachment_name=None):
    """Send email using Gmail API"""
    try:
        service = get_gmail_service()
        
        # Create message
        message = MIMEMultipart()
        message['to'] = recipient_email
        message['subject'] = subject
        
        # Add text body
        msg_text = MIMEText(body, 'html')
        message.attach(msg_text)
        
        # Add attachment if provided
        if attachment:
            attachment_mime = MIMEApplication(attachment.getvalue(), _subtype='pdf')
            attachment_mime.add_header('Content-Disposition', 'attachment', filename=attachment_name)
            message.attach(attachment_mime)
            
        # Encode message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        # Send message
        send_message = service.users().messages().send(
            userId= recipient_email, 
            body={'raw': encoded_message}
        ).execute()
        
        return True, f"Email sent successfully. Message ID: {send_message['id']}"
    
    except Exception as e:
        return False, f"Error sending email: {str(e)}"


st.header("Advanced AI-Driven Legal Document Summarization and Risk Assessment")

# Add helpful instructions
with st.expander("How to use this chat"):
    st.markdown("""
    1. Upload your legal documents using the sidebar
    2. Click 'Generate Summary' to print summary
    3. Click 'Generate Risk Assessment' to analyze risks
    4. Ask questions about your documents in the chat
    5. Enter your email to receive reports directly
    6. The AI will maintain context throughout the conversation
    """)

# Initialize session state variables
if any(key not in st.session_state for key in ["summary", "risk_assessment", "conversation", "chat_history", "vectorstore", "processed", "risk_percentage", "user_email"]):
    st.session_state.update({
        "summary": None,
        "risk_assessment": None,
        "conversation": None,
        "chat_history": [],
        "vectorstore": None,
        "processed": False,
        "risk_percentage": None,
        "user_email": ""
    })

def process_pdfs(uploaded_files):
    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        future_text = executor.submit(extract_text_from_pdfs, uploaded_files)
        raw_text = future_text.result()
        chunks = split_text_into_chunks(raw_text)
        st.session_state.vectorstore = create_vectorstore(chunks)

def extract_text_from_pdfs(pdf_docs):
    """Extract text from PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text):
    """Split text into chunks"""
    text_splitters = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=50)
    return text_splitters.split_text(text)

def create_vectorstore(chunks):
    """Create and return a vector store from document chunks"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def get_conversation_chain():
    """Create a conversation chain with memory and retrieval from vector store"""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(),
        memory=memory
    )

class IndianRegulatoryInsights:
    def __init__(self):
        # Public APIs and data sources for Indian government information
        self._api_sources = {
            "INDIA_OPEN_GOVERNMENT_DATA": "https://data.gov.in/api/1/dataset",
            "LEGISLATIVE_DEPT_GOV": "https://legislative.gov.in/api/acts",
            "NATIONAL_PORTAL_API": "https://digitalindia.gov.in/api/public-insights"
        }
        
        # API headers to ensure proper request formatting
        self._headers = {
            "User-Agent": "LegalDocumentAssistant/1.0",
            "Accept": "application/json"
        }
    
    def fetch_regulatory_insights(self, topic="data_protection"):
        """
        Fetch regulatory insights from Indian government sources
        """
        insights = {
            "source": "Indian Government Open Data",
            "timestamp": datetime.now().isoformat(),
            "results": []
        }
        
        try:
            # Primary source: data.gov.in open data portal
            response = requests.get(
                self._api_sources["INDIA_OPEN_GOVERNMENT_DATA"],
                params={
                    "filters[topic]": topic,
                    "sort_by": "created_date",
                    "format": "json",
                    "max_results": 5
                },
                headers=self._headers
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Process results from government open data
                if 'results' in data:
                    for result in data['results']:
                        insights['results'].append({
                            "title": result.get('title', 'Untitled Document'),
                            "description": result.get('description', 'No description available'),
                            "department": result.get('department', 'Unknown'),
                            "published_date": result.get('created_date', 'N/A'),
                            "url": result.get('download_url', '')
                        })
            
            # Attempt to fetch legislative acts as supplementary information
            if not insights['results']:
                leg_response = requests.get(
                    self._api_sources["LEGISLATIVE_DEPT_GOV"],
                    params={
                        "keyword": topic,
                        "limit": 3
                    },
                    headers=self._headers
                )
                
                if leg_response.status_code == 200:
                    leg_data = leg_response.json()
                    for act in leg_data.get('acts', []):
                        insights['results'].append({
                            "title": act.get('name', 'Unnamed Act'),
                            "description": act.get('summary', 'No summary available'),
                            "year": act.get('year', 'N/A'),
                            "url": f"https://legislative.gov.in/acts/{act.get('id')}"
                        })
            
            return insights
        
        except requests.RequestException as e:
            st.error(f"Error fetching Indian regulatory insights: {e}")
            return {
                "error": "Unable to fetch regulatory insights",
            }
    def get_legal_resources(self):
        """
        Compile list of key legal resources and regulatory bodies
        """
        legal_resources = [
            {
                "name": "Ministry of Electronics and Information Technology (MeitY)",
                "description": "Responsible for digital governance and data protection policies",
                "website": "https://www.meity.gov.in/"
            },
            {
                "name": "National Informatics Centre (NIC)",
                "description": "Provides technological support for government digital initiatives",
                "website": "https://www.nic.in/"
            }
        ]
        return legal_resources    
def assess_risks(text):
    """Assess risks from the text and return a dictionary of risk areas with scores out of 10"""
    # Define risk areas and associated keywords
    risk_areas = {
        "Financial Risk": ["financial", "loss", "budget", "cost"],
        "Operational Risk": ["operational", "process", "failure", "delay"],
        "Compliance Risk": ["compliance", "regulation", "legal", "law"],
        "Reputation Risk": ["reputation", "brand", "image", "public"],
        "Strategic Risk": ["strategy", "market", "competition", "trend"]
    }

    # Initialize risk scores
    risk_scores = {area: 0 for area in risk_areas}

    # Calculate risk scores based on keyword occurrences
    for area, keywords in risk_areas.items():
        score = sum(text.lower().count(keyword) for keyword in keywords)
        # Normalize score to be out of 10
        risk_scores[area] = min(score, 10)

    return risk_scores

def plot_risks(risk_scores):
    """Plot the risk scores as a pie chart"""
    labels = risk_scores.keys()
    sizes = risk_scores.values()

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

def generate_summary():
    """Generate summary using LLM and Vector Store"""
    summary_prompt = """
    Provide a concise summary of the document focusing on key points and main findings.
    """
    try:
        if st.session_state.vectorstore is None:
            st.error("Vector store is not initialized. Please upload and process PDFs first.")
            return None

        relevant_docs = st.session_state.vectorstore.similarity_search(summary_prompt, k=2)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        final_prompt = f"{summary_prompt}\n\nDocument context: {context}"
        return llm.predict(final_prompt)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def generate_risk_assessment():
    """Generate risk assessment using LLM and keyword checking"""
    risk_prompt = """
    Identify potential legal risks in the document. Focus on high-risk areas and compliance issues.
    """
    risk_keywords = ["risk", "liability", "compliance", "exposure", "legal issue"]
    try:
        if st.session_state.vectorstore is None:
            st.error("Vector store is not initialized. Please upload and process PDFs first.")
            return None

        relevant_docs = st.session_state.vectorstore.similarity_search(risk_prompt, k=2)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        final_prompt = f"{risk_prompt}\n\nDocument context: {context}"
        return llm.predict(final_prompt)
    except Exception as e:
        st.error(f"Error generating risk assessment: {e}")
        return None

def create_pdf(content, title):
    """Create a PDF with proper text wrapping and pagination"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["BodyText"]),
        Spacer(1, 12)
    ]
    story.extend(Paragraph(line, styles["BodyText"]) for line in content.split("\n"))
    doc.build(story)
    buffer.seek(0)
    return buffer

def handle_user_question(user_question):
    """Process user question and get response from the conversation chain"""
    response = st.session_state.conversation({"question": user_question, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((user_question, response["answer"]))
    return response["answer"]

def create_email_html(summary, risk_assessment=None):
    """Create HTML email content with summary and risk assessment"""
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: grey; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
            h2 {{ color: blue; margin-top: 20px; }}
            .footer {{ margin-top: 30px; font-size: 12px; color: #7f8c8d; text-align: center; }}
            .risk-high {{ color: #e74c3c; }}
            .risk-medium {{ color: #f39c12; }}
            .risk-low {{ color: #27ae60; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Legal Document Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Document Summary</h2>
            <div>
                {summary.replace('\n', '<br>')}
            </div>
    """
    
    if risk_assessment:
        html += f"""
            <h2>Risk Assessment</h2>
            <div>
                {risk_assessment.replace('\n', '<br>')}
            </div>
        """
    
    html += """
            <div class="footer">
                <p>This report was generated by AI-Driven Legal Document Analysis System.</p>
                <p>Please review the information with appropriate legal counsel.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

# Add consent mechanism
st.sidebar.checkbox("I consent to the processing of my data", value=True, key="gdpr_consent")

# Add data management options
if st.sidebar.button("Request Data Deletion"):
    st.session_state.clear()
    st.success("Your data has been deleted.")

# Sidebar for document upload and email setup
with st.sidebar:
    st.subheader("Your Docs")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True)
    
    st.subheader("Email Reports")
    st.session_state.user_email = st.text_input("Your Email Address", value=st.session_state.user_email)

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Risk Assessment", "Chat", "🇮🇳 Regulatory Insights", "📧 Email Reports"])

with tab1:
    st.header("Document Summary")
    if st.button("Generate Summary"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                process_pdfs(uploaded_files)
            with st.spinner("Generating summary..."):
                summary = generate_summary()
                if summary:
                    st.session_state.summary = summary
                    st.success("Summary generated successfully!")
        else:
            st.error("Please upload PDF files first.")
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
        pdf_summary = create_pdf(st.session_state.summary, "Document Summary")
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
                risk_assessment = generate_risk_assessment()
                if risk_assessment:
                    st.session_state.risk_assessment = risk_assessment
                    st.success("Risk assessment generated successfully!")
                    raw_text = extract_text_from_pdfs(uploaded_files)
                    risk_scores = assess_risks(raw_text)
                    plot_risks(risk_scores)
        else:
            st.error("Please upload PDF files first.")
    if st.session_state.risk_assessment:
        st.markdown(st.session_state.risk_assessment)
        pdf_risk = create_pdf(st.session_state.risk_assessment, "Risk Assessment Report")
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
                st.session_state.conversation = get_conversation_chain()
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

with tab4:  
    # New tab for Indian Regulatory Insights
    st.header("Indian Regulatory Insights")
    
    # Create an instance of IndianRegulatoryInsights
    insights_manager = IndianRegulatoryInsights()
    
    # Add a selectbox for choosing regulatory topics
    insight_topic = st.selectbox(
        "Select Regulatory Topic",
        [
            "data_protection", 
            "digital_privacy", 
            "information_technology", 
            "cyber_law", 
            "government_regulations"
        ]
    )
    
    # Create a button to fetch insights
    if st.button("Fetch Indian Regulatory Information"):
        if st.session_state.get("gdpr_consent", False):
            with st.spinner("Retrieving Indian regulatory insights..."):
                insights = insights_manager.fetch_regulatory_insights(insight_topic)
                
                if insights and 'results' in insights:
                    st.success(f"Insights from {insights.get('source', 'Indian Government Sources')}")
                    st.write(f"Retrieved at: {insights.get('timestamp')}")

                else:
                    st.warning("No insights available for the selected topic.")
        else:
            st.error("User consent is required to fetch regulatory insights.")
    
    # Additional legal resources section
    if st.button("View Key Legal Resources"):
        legal_resources = insights_manager.get_legal_resources()
        st.subheader("Key Indian Legal and Regulatory Bodies")
        
        for resource in legal_resources:
            with st.expander(resource['name']):
                st.write(resource['description'])
                st.markdown(f"**Website:** [{resource['website']}]({resource['website']})")

# New tab for Email Reports
with tab5:
    st.header("Email Analysis Reports")
    
    email_col1, email_col2 = st.columns(2)
    
    with email_col1:
        st.subheader("Email Summary Report")
        if st.button("Send Summary Report"):
            if not st.session_state.user_email:
                st.error("Please enter your email address in the sidebar.")
            elif not st.session_state.summary:
                st.error("Please generate a document summary first.")
            else:
                with st.spinner("Sending summary report..."):
                    # Create PDF attachment
                    pdf_buffer = create_pdf(st.session_state.summary, "Document Summary")
                    
                    # Create email content
                    email_html = create_email_html(st.session_state.summary)
                    
                    # Send email
                    success, message = send_email(
                        st.session_state.user_email,
                        "Your Legal Document Summary Report",
                        email_html,
                        pdf_buffer,
                        "document_summary.pdf"
                    )
                    
                    if success:
                        st.success("Summary report sent successfully!")
                    else:
                        st.error(message)
    
    with email_col2:
        st.subheader("Email Complete Analysis")
        if st.button("Send Complete Analysis"):
            if not st.session_state.user_email:
                st.error("Please enter your email address in the sidebar.")
            elif not st.session_state.summary or not st.session_state.risk_assessment:
                st.error("Please generate both summary and risk assessment first.")
            else:
                with st.spinner("Sending complete analysis..."):
                    # Create combined PDF
                    combined_content = f"DOCUMENT SUMMARY\n\n{st.session_state.summary}\n\n"
                    combined_content += f"RISK ASSESSMENT\n\n{st.session_state.risk_assessment}"
                    pdf_buffer = create_pdf(combined_content, "Legal Document Complete Analysis")
                    
                    # Create email content with both sections
                    email_html = create_email_html(
                        st.session_state.summary,
                        st.session_state.risk_assessment
                    )
                    
                    # Send email
                    success, message = send_email(
                        st.session_state.user_email,
                        "Your Complete Legal Document Analysis",
                        email_html,
                        pdf_buffer,
                        "complete_legal_analysis.pdf"
                    )
                    
                    if success:
                        st.success("Complete analysis sent successfully!")
                    else:
                        st.error(message)
    
    st.markdown("""
    ### Email Delivery Status
    
    Once you request an email to be sent, you'll see a confirmation here. 
    Emails include:
    
    - PDF attachments of the requested reports
    - Formatted HTML content for easy reading
    - Summary of key findings
    - Date and time of analysis
    
    **Note:** Please make sure your email address is correct before sending any reports.
    """)
    
    # Email templates preview
    with st.expander("Preview Email Template"):
        st.image("https://via.placeholder.com/800x500?text=Email+Template+Preview", use_column_width=True)
        st.caption("Example of how your email will appear with section formatting and branding.")