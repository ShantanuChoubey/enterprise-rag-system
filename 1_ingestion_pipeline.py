
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory."""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8-sig"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in {docs_path}."
        )

    print(f"Loaded {len(documents)} documents.")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap."""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks



def create_vector_store(chunks, persist_directory="db/chroma_ollama"):
    """Create and persist ChromaDB vector store in batches."""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # Create an empty persistent vector store.
    vectorstore = Chroma(
        collection_name="rag_documents",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Add small batches to avoid one large embedding request.
    batch_size = 10

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectorstore.add_documents(batch)

        completed = min(start + len(batch), len(chunks))
        print(f"Embedded and stored {completed}/{len(chunks)} chunks")

    print("--- Finished creating vector store ---")
    print(f"Vector store saved to {persist_directory}")
    print(f"Stored chunks: {vectorstore._collection.count()}")

    return vectorstore



def main():
    """Main ingestion pipeline."""
    print("=== RAG Document Ingestion Pipeline ===\n")

    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_ollama"

    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("Checking existing Ollama vector store...")

        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(
        collection_name="rag_documents",
        persist_directory=persistent_directory,
        embedding_function=embedding_model
        )

        chunk_count = vectorstore._collection.count()

        if chunk_count > 0:
            print(
                f"Loaded existing vector store with "
                f"{chunk_count} chunks"
            )
            return vectorstore

        raise RuntimeError(
            f"The vector store at '{persistent_directory}' "
            "exists but contains no chunks. Inspect or remove "
            "this incomplete database before retrying."
        )

    print("Persistent directory does not exist. Initializing vector store...\n")

    # Step 1: Load documents
    documents = load_documents(docs_path)

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Create vector store
    vectorstore = create_vector_store(chunks, persistent_directory)

    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore


if __name__ == "__main__":
    main()


# Sample Document objects for learning (commented out)
# from langchain_core.documents import Document
#
# documents = [
#     Document(
#         page_content="Google LLC is an American multinational corporation and technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).",
#         metadata={'source': 'docs/google.txt'}
#     ),
#     Document(
#         page_content="Microsoft Corporation is an American multinational corporation and technology conglomerate headquartered in Redmond, Washington.",
#         metadata={'source': 'docs/microsoft.txt'}
#     ),
#     Document(
#         page_content="Nvidia Corporation is an American technology company headquartered in Santa Clara, California.",
#         metadata={'source': 'docs/nvidia.txt'}
#     ),
#     Document(
#         page_content="Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American space technology company headquartered at the Starbase development site in Starbase, Texas.",
#         metadata={'source': 'docs/spacex.txt'}
#     ),
#     Document(
#         page_content="Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas.",
#         metadata={'source': 'docs/tesla.txt'}
#     )
# ]
