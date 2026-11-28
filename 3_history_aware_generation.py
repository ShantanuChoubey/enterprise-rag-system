from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Connect to the existing local ChromaDB vector store
persistent_directory = "db/chroma_ollama"
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    collection_name="rag_documents",
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)

# Initialize local Ollama LLM (temperature=0 prevents query-rewrite drift/hallucination)
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

# Conversation memory
chat_history = []


def reformulate_question(user_question: str) -> str:
    """Rewrite follow-up questions into standalone search queries using chat history."""
    if not chat_history:
        return user_question

    rewrite_messages = [
        SystemMessage(
            content=(
                "Given the chat history and the latest user question, rewrite the latest question "
                "into a single standalone search query that can be understood without the chat history. "
                "Do NOT answer the question. Output ONLY the rewritten standalone question and nothing else."
            )
        ),
        *chat_history[-6:],  # Keep last 3 turns (6 messages) for fast, focused local context
        HumanMessage(content=f"Latest question: {user_question}")
    ]

    result = llm.invoke(rewrite_messages)
    rewritten = result.content.strip().strip('"')
    return rewritten if rewritten else user_question


def ask_question(user_question: str) -> str:
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Reformulate follow-up question if history exists
    search_question = reformulate_question(user_question)
    if search_question != user_question:
        print(f"Reformulated Search Query: {search_question}")

    # Step 2: Retrieve relevant chunks from ChromaDB
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(search_question)

    print(f"Retrieved {len(docs)} relevant chunks:")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        preview = doc.page_content.replace("\n", " ")[:100]
        print(f"  [{i}] {source} (Page {page}): {preview}...")

    # Step 3: Build grounded context with source filenames
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')} (Page: {doc.metadata.get('page', 'N/A')})\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )

    combined_input = f"""Based on the following retrieved context, answer the user's question.

Rules:
1. Answer using only the provided context.
2. Give a clear, direct answer.
3. Do not invent facts or add unsupported information.
4. If the context does not contain the answer, say: "I could not find enough information in the documents."
5. Mention the source filename when possible.

Context:
{context}

Question:
{user_question}

Answer:"""

    # Step 4: Generate grounded response
    messages = [
        SystemMessage(content="You are a strict enterprise RAG assistant that answers only from retrieved documents."),
        *chat_history[-6:],
        HumanMessage(content=combined_input)
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()

    # Step 5: Save turn to conversation history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"\nAnswer:\n{answer}")
    return answer


def start_chat():
    print("=== History-Aware Enterprise RAG Chatbot (Ollama Local) ===")
    print("Ask follow-up questions freely! Type 'quit' or 'exit' to stop.\n")

    while True:
        question = input("Your question: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        ask_question(question)


if __name__ == "__main__":
    start_chat()