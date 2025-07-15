import streamlit as st
from vector_store import get_vector_store
from data_loader import load_wikipedia
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# --- Page Config ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# --- Title ---
st.markdown("<h1 style='text-align: center;'>RAG Chatbot</h1>", unsafe_allow_html=True)

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- User Input ---
query = st.text_input("Ask your question here:", placeholder="e.g. How does machine learning work?")

# --- Processing ---
if query:
    st.session_state.messages.append(("user", query))
    with st.spinner("Retrieving information..."):
        try:
            docs = load_wikipedia(query)
            vectorstore = get_vector_store(docs, query)
            llm = Ollama(model="llama3")
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
            answer = qa.invoke({"query": query})
        except Exception as e:
            answer = f"Error: {str(e)}"
    st.session_state.messages.append(("bot", answer))

# --- Chat Display ---
for role, message in st.session_state.messages:
    if role == "user":
        st.markdown(f"<div style='padding:10px 15px; background-color:#f0f2f6; border-radius:10px; margin-bottom:10px;'><strong>You:</strong> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='padding:10px 15px; background-color:#e8f1ea; border-radius:10px; margin-bottom:20px;'><strong>Bot:</strong> {message}</div>", unsafe_allow_html=True)

# --- Sidebar Info ---
with st.sidebar:
    st.header("Chat History")
    for i, (role, msg) in enumerate(st.session_state.messages):
        if role == "user":
            st.markdown(f"- {msg[:60]}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()
