import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Initialize the Hugging Face embedding model (local, no API required)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vector_store(docs, query):
    # If it's a plain string, wrap in a Document
    if isinstance(docs, str):
        docs = [Document(page_content=docs)]

    # Extract valid page content
    texts = [doc.page_content for doc in docs if doc.page_content.strip()]
    if not texts:
        raise ValueError("❌ No valid text extracted from input.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(texts)
    content = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
    if not content:
        raise ValueError("❌ Text splitter returned no valid chunks.")

    print(f"[DEBUG] {len(content)} chunks ready for FAISS.")

    # Cache path setup (use sanitized query)
    safe_query = "".join(c for c in query if c.isalnum() or c in (" ", "_")).replace(" ", "_").strip()
    path = os.path.join("faiss_cache", safe_query)

    # If the FAISS index exists, load it
    if os.path.exists(os.path.join(path, "index.faiss")):
        print("[INFO] Loading cached FAISS index...")
        return FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)

    # Else: Create and cache a new index
    print("[INFO] Creating new FAISS index...")
    os.makedirs(path, exist_ok=True)
    vectorstore = FAISS.from_texts(content, embedder)
    vectorstore.save_local(path)
    return vectorstore


