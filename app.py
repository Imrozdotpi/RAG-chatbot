import streamlit as st
from vector_store import get_vector_store
from data_loader import load_wikipedia
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory

# --- Page Config ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# --- Title ---
st.markdown("<h1 style='text-align: center;'>RAG Chatbot</h1>", unsafe_allow_html=True)

# --- Session State Initialization ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_archive" not in st.session_state:
    st.session_state.chat_archive = []

# --- User Input ---
query = st.text_input("Ask your question here:", placeholder="e.g. How does machine learning work?")

# --- Process Query ---
if query:
    st.session_state.messages.append(("user", query))
    with st.spinner("Retrieving information..."):
        try:
            if st.session_state.chain is None:
                docs = load_wikipedia(query)
                vectorstore = get_vector_store(docs, query)
                llm = Ollama(model="llama3")
                st.session_state.chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                    memory=st.session_state.memory,
                    verbose=False
                )
            result = st.session_state.chain.invoke({"question": query})
            answer = result["answer"]
        except Exception as e:
            answer = f"Error: {str(e)}"
    st.session_state.messages.append(("bot", answer))

# --- Chat Display ---
for role, message in st.session_state.messages:
    bg_color = "#f2f2f2" if role == "user" else "#e6e6e6"
    name = "You" if role == "user" else "Bot"
    st.markdown(
        f"<div style='padding:12px 16px; background-color:{bg_color}; border-radius:8px; margin-bottom:10px;'>"
        f"<strong>{name}:</strong> {message}</div>", unsafe_allow_html=True
    )

# --- Sidebar Chat History ---
with st.sidebar:
    st.header("Chat History")
    for role, msg in st.session_state.messages:
        if role == "user":
            st.markdown(f"- {msg[:60]}")

    st.subheader("Archived Sessions")
    for i, chat in enumerate(st.session_state.chat_archive):
        st.markdown(f"**Session {i+1}:**")
        for role, msg in chat:
            if role == "user":
                st.markdown(f"- {msg[:50]}")

    # --- Minimal Gray Button ---
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #f1f1f1;
            color: #333;
            border: 1px solid #ccc;
            border-radius: 6px;
            padding: 0.45rem 1rem;
            font-size: 14px;
            width: 100%;
            transition: background-color 0.2s ease;
        }
        .stButton > button:hover {
            background-color: #e2e2e2;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button("Clear Chat"):
        if st.session_state.messages:
            st.session_state.chat_archive.append(st.session_state.messages.copy())
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.chain = None
        st.experimental_rerun()
