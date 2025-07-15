from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from vector_store import get_vector_store
from data_loader import load_wikipedia

def query_engine(vectorstore, question):
    llm = Ollama(model="llama3")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    return qa.invoke({"query": question})

if __name__ == "__main__":
    question = input("Enter your question: ")
    wiki_content = load_wikipedia(question)
    vectorstore = get_vector_store(wiki_content, question)
    print("\n[INFO] Generating answer...\n")
    answer = query_engine(vectorstore, question)
    print("🧠 Answer:\n", answer)
