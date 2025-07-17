from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory

from vector_store import get_vector_store
from data_loader import load_wikipedia

def main():
    print("\n🧠 Welcome to your RAG ChatBot with memory! Type 'exit' to quit.\n")

    # Set topic & get Wikipedia content
    topic = input("📌 What topic would you like to explore? ")
    docs = load_wikipedia(topic)
    vectorstore = get_vector_store(docs, topic)

    # Initialize LLM + memory
    llm = Ollama(model="llama3")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Build conversational retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=False
    )

    # Chat loop
    while True:
        question = input("\n🧑 You: ")
        if question.lower() in ["exit", "quit"]:
            print("👋 Exiting. Bye!")
            break

        result = qa_chain.invoke({"question": question})
        print("\n🤖 Bot:", result["answer"])

if __name__ == "__main__":
    main()
