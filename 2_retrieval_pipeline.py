
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

# Connect to the existing vector store
persistent_directory = "db/chroma_ollama"
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    collection_name="rag_documents",
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)

# Initialize the local language model
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

# Get the user's question
query = "In what year did Tesla begin production of the Roadster?"

# Retrieve relevant document chunks
retriever = db.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
print("\n--- Retrieved Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(doc.page_content)

# Combine retrieved chunks into context
context = "\n\n".join(
    f"Source: {doc.metadata.get('source', 'Unknown')}\n"
    f"Content: {doc.page_content}"
    for doc in relevant_docs
)

# Prepare the prompt for answer generation
prompt = f"""
You are an assistant answering questions using retrieved documents.

Rules:
1. Answer using only the provided context.
2. Give a clear, direct answer.
3. Do not invent facts or add unsupported information.
4. If the context does not contain the answer, say:
   "I could not find enough information in the documents."
5. Mention the source filename when possible.

Context:
{context}

Question:
{query}

Answer:
"""

# Generate the final answer
print("\n--- Generated Answer ---")

response = llm.invoke(prompt)
print(response.content)




# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
