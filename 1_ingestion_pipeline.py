import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all .txt and .pdf files from the docs directory."""
    print(f"Loading documents from '{docs_path}'...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory '{docs_path}' does not exist."
        )

    documents = []
    for root, _, files in os.walk(docs_path):
        for filename in sorted(files):
            file_path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()

            try:
                if ext == ".txt":
                    loader = TextLoader(file_path, encoding="utf-8-sig")
                    loaded = loader.load()
                    for doc in loaded:
                        doc.metadata["source"] = filename
                        doc.metadata["file_type"] = "txt"
                    documents.extend(loaded)
                elif ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                    loaded = loader.load()
                    for doc in loaded:
                        doc.metadata["source"] = filename
                        doc.metadata["file_type"] = "pdf"
                    documents.extend(loaded)
            except Exception as e:
                print(f"Warning: Failed to load '{filename}': {e}")

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in '{docs_path}'."
        )

    print(f"Loaded {len(documents)} document pages/files.")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks with recursive structure awareness."""
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata.get('source', 'Unknown')} (Page: {chunk.metadata.get('page', 'N/A')})")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Preview: {chunk.page_content[:150]}...")
            print("-" * 50)

        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_ollama"):
    """Create and persist ChromaDB vector store in batches using local Ollama."""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma(
        collection_name="rag_documents",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

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

    docs_path = "docs"
    persistent_directory = "db/chroma_ollama"

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
            print(f"Loaded existing vector store with {chunk_count} chunks")
            return vectorstore

        raise RuntimeError(
            f"The vector store at '{persistent_directory}' exists but contains no chunks. "
            "Remove this incomplete database before retrying."
        )

    print("Persistent directory does not exist. Initializing vector store...\n")

    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks, persistent_directory)

    print("\nIngestion complete! Your documents are now ready for RAG queries.")
    return vectorstore


if __name__ == "__main__":
    main()