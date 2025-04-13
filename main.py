import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
# groq_api_key = st.secrets.get("GROQ_API_KEY")
# hf_token = st.secrets.get("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token

# === Validate keys ===
if not groq_api_key or not hf_token:
    st.error("Missing API keys. Please add GROQ_API_KEY and HF_TOKEN to your .env.")
    st.stop()

# === Embeddings & LLM setup ===
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

# === Chat history memory ===
store = {}
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in store:
        store[session] = ChatMessageHistory()
    return store[session]

# === Streamlit page settings ===
st.set_page_config(page_title="Chat with PDFs", layout="wide")
st.markdown("<h2 style='text-align:center;'>PDF Q&A Chatbot</h2>", unsafe_allow_html=True)
st.caption("Ask questions directly based on your uploaded documents.")

# === Session state ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# === Sidebar (mobile responsive) ===
with st.sidebar:
    st.subheader("📂 Upload PDFs")
    uploaded_files = st.file_uploader("Select PDF(s)", type="pdf", accept_multiple_files=True)

    if st.button("Start New Conversation"):
        st.session_state.chat_history = []
        st.session_state.rag_chain = None
        st.session_state.uploaded_files = []
        st.rerun()

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

    with st.expander("How it works"):
        st.markdown("""
        - Upload one or more PDF files.
        - Ask questions based on their content.
        - To start new conversation first remove current pdf and click on "Start New Conversation"
        """)

# === RAG Chain Setup ===
@st.cache_resource(show_spinner="Analyzing and indexing your documents...")
def setup_rag_chain_from_upload(files):
    documents = []
    for uploaded_file in files:
        if uploaded_file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Given a chat history and the latest user question "
         "which might reference context in the chat history, "
         "formulate a standalone question which can be understood "
         "without the chat history. Do NOT answer the question, "
         "just reformulate it if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are a helpful assistant answering questions about documents. "
        "Use the following retrieved context to answer the user's question. "
        "\n\n{context}\n\n"
        "Instructions:\n"
        "1. Answer using the document context.\n"
        "2. If no context is found, say so.\n"
        "3. If unclear, ask the user for clarification."
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

# === Initialize chain if files uploaded ===
if st.session_state.uploaded_files and not st.session_state.rag_chain:
    st.session_state.rag_chain = setup_rag_chain_from_upload(st.session_state.uploaded_files)

# === Main Chat Section ===
if st.session_state.uploaded_files:
    with st.chat_message("assistant"):
        st.markdown("PDFs loaded. Ask your questions below!")

    user_input = st.chat_input("Type your question...")

    if user_input:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": "default_session"}}
                )
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", response["answer"]))
            except Exception as e:
                st.error(f"Error: {e}")

    # Display history
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

else:
    st.info("Upload at least one PDF from the sidebar to begin.")

