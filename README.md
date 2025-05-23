# PDF Question-Answer Chatbot
The Project provides a comprehensive Streamlit-based(UI) application that enables users to seamlessly upload PDF documents and interact with their content through intuitive question-answering capabilities. This intelligent chatbot leverages LangChain, FAISS, Hugging Face Embeddings, GROQ LLM, and RAG (Retrieval Augmented Generation) to deliver precise, context-aware answers derived directly from the uploaded document content.

## Project Structure
PDF-Question-Answer-Chatbot/
   1) ├── .gitignore
   2) ├── chatbot.ipynb             # Step-by-step notebook version of the chatbot
   3) ├── main.py                   # Production-ready Streamlit app
   4) ├── requirements.txt          # Project dependencies
   5) ├── README.md                 # Project documentation
   6) └── pdfenv/                   # Conda environment folder (ignored by git)

## Features
   1) Upload multiple PDF files simultaneously
   2) Ask questions directly related to document content
   3) Context-aware responses through RAG (Retrieval Augmented Generation)
   4) Persistent chat history throughout the session
   5) Simple and intuitive user interface

## Architecture:
The application uses a simple and sophisticated architecture combining several key technologies:
   1) Document Processing: PDFs are loaded, processed, and split into manageable chunks
   2) Embeddings: Document chunks are converted to vector embeddings using Hugging Face's all-MiniLM-L6-v2 model
   3) Vector Storage: FAISS is used for efficient similarity search
   4) Retrieval: Context-aware retriever that considers chat history
   5) LLM: Leverages Groq's llama3-70b-8192 model for intelligent responses
   6) UI: Streamlit for an intuitive user interface

## Tech Stack
   1) Frontend/UI: Streamlit
   2) Language Model: Groq (llama3-70b-8192)
   3) Embeddings: Hugging Face (all-MiniLM-L6-v2)
   4) Vector Database: FAISS
   5) Document Processing: LangChain
   6) PDF Processing: PyPDF
   7) programming language: python==3.12 (Prerequisite)

### The Chatbot Application is Deployed on Streamlit Clouds, can be accessed via link
https://pdf-question-answer-chatbot-sxmvssqson4b4sg5dgrftd.streamlit.app
