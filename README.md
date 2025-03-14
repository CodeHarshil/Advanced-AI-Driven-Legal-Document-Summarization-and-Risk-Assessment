
# AI-Powered Legal Document Summarization and Risk Identification

## Project Overview
This application allows us to upload multiple PDF documents and chat with them using LLaMA through Groq's API. The application uses Streamlit for the interface, Langchain for the document processing pipeline, HuggingFace for embeddings and FAISS for vector database.

## Key Components

- **Document Processing Engine**: Converts PDF/Word files to structured text and identifies key sections.
- **LLM-Integrated Analysis Core**: Incorporates RAG for summarization and risk detection.
- **Compliance Validation Layer**: Cross-references terms against regulatory frameworks.
- **Centralized Dashboard**: Streamlined interface with investigation tools and version control.

## Features

1. 📄 **Upload Your Documents**: Easily upload legal documents for analysis.
2. 📃 **Generates Summary and Risk Assessment**: Provides a summary and risk assessment of the uploaded file.
3. 🗨️ **Chat with Your Documents**: Interact with your documents through a chatbot interface.
4. 🔗 **Compliance Websites**: Provides links to compliance websites related to risk areas.
5. 📩 **Email Notifications**: Send summary and risk assessments to the user's email.

## How to Use

1. Upload your legal documents through the interface.
2. View the generated summary and risk assessment.
3. Use the chat feature to interact with your documents.
4. Access compliance-related websites for further information.
5. Opt to receive summaries and risk assessments via email.

## Technologies Used

- Python for backend processing
- Streamlit for the user interface
- Groq and Hugging Face for LLM integration
- SendGrid for email automation
