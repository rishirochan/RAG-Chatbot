import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

# Define Project Roots
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) # Go up from RAG_Dev to root
DOCS_DIR = os.path.join(PROJECT_ROOT, "documents")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "rag_vector_store")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# Load environment variables
load_dotenv(ENV_PATH)

def get_llm():
    """Initialize and return the Groq LLM."""
    # Ensure env is loaded
    load_dotenv(ENV_PATH)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(f"GROQ_API_KEY not found. Checked: {ENV_PATH}")
    return ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

def get_embeddings():
    """Initialize and return the Ollama embeddings model."""
    return OllamaEmbeddings(model="nomic-embed-text")

def load_vector_store(persist_dir=VECTOR_STORE_DIR):
    """
    Attempt to load an existing Chroma vector store from disk.
    Returns the vector store if successful, else None.
    """
    if os.path.exists(persist_dir):
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=get_embeddings(),
            collection_metadata={"hnsw:space": "cosine"}
        )
        return vector_store
    return None

def create_vector_store(doc_dir=DOCS_DIR, persist_dir=VECTOR_STORE_DIR):
    """
    Create a new Chroma vector store from PDFs in the specified directory.
    Returns the new vector store.
    """
    loaders = PyPDFDirectoryLoader(doc_dir)
    documents = loaders.load()
    
    if not documents:
        return None, 0
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)
    
    vector_store = Chroma.from_documents(
        documents=final_documents,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return vector_store, len(final_documents)

def add_documents_to_store(doc_dir, vector_store):
    """
    Load PDFs from doc_dir, split them, and add to the existing vector_store.
    Returns the number of chunks added.
    """
    loaders = PyPDFDirectoryLoader(doc_dir)
    documents = loaders.load()
    
    if not documents:
        return 0
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)
    
    vector_store.add_documents(final_documents)
    return len(final_documents)

def get_indexed_documents(vector_store):
    """
    Get list of unique document filenames from the vector store.
    Returns a list of document filenames.
    """
    try:
        collection = vector_store._collection
        existing_data = collection.get(include=["metadatas"])
        sources = set()
        for meta in existing_data.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(os.path.basename(meta["source"]))
        return sorted(list(sources))
    except:
        return []

def delete_documents_from_store(filenames_to_delete, vector_store, doc_dir=DOCS_DIR):
    """
    Delete documents from the vector store by filename.
    Also removes the PDF files from disk.
    Returns tuple: (num_deleted_chunks, num_deleted_files)
    """
    deleted_chunks = 0
    deleted_files = 0
    
    try:
        collection = vector_store._collection
        
        # Get all documents from collection
        all_data = collection.get(include=["metadatas"])
        
        for filename in filenames_to_delete:
            ids_to_delete = []
            for i, meta in enumerate(all_data.get("metadatas", [])):
                if meta and "source" in meta:
                    # Match by basename (filename only)
                    if os.path.basename(meta["source"]) == filename:
                        ids_to_delete.append(all_data["ids"][i])
            
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                deleted_chunks += len(ids_to_delete)
            
            # Delete PDF from disk
            file_path = os.path.join(doc_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files += 1
    except Exception as e:
        print(f"Error deleting documents: {e}")
    
    return deleted_chunks, deleted_files

def add_files_to_store(file_paths, vector_store):
    """
    Load specific PDF files, split them, and add to the existing vector_store.
    Skips files that are already indexed (checks by filename).
    Returns tuple: (chunks_added, files_skipped)
    """
    # Get already indexed filenames from vector store metadata
    try:
        collection = vector_store._collection
        existing_data = collection.get(include=["metadatas"])
        indexed_sources = set()
        for meta in existing_data.get("metadatas", []):
            if meta and "source" in meta:
                indexed_sources.add(os.path.basename(meta["source"]))
    except:
        indexed_sources = set()
    
    all_documents = []
    skipped = 0
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if filename in indexed_sources:
            skipped += 1
            continue
        loader = PyPDFLoader(file_path)
        all_documents.extend(loader.load())
    
    if not all_documents:
        return 0, skipped
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(all_documents)
    
    vector_store.add_documents(final_documents)
    return len(final_documents), skipped

def get_rag_chain_response(vector_store, question):
    """
    Create the RAG chain and invoke it with the question.
    Returns the full response dictionary.
    """
    llm = get_llm()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    
    # Format context from documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt and get response
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based on the context provided only. Be accurate and concise."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    return {
        "answer": response.content,
        "context": docs
    }

def get_similarity_scores(vector_store, question, k=10):
    """
    Perform similarity search with scores and deduplicate results.
    Returns a list of tuples (document, score) with unique content.
    """
    results = vector_store.similarity_search_with_relevance_scores(question, k=k)
    
    # Deduplicate by content (keep first occurrence = highest score)
    seen_content = set()
    unique_results = []
    for doc, score in results:
        # Use first 200 chars as fingerprint to catch near-duplicates
        fingerprint = doc.page_content[:200].strip()
        if fingerprint not in seen_content:
            seen_content.add(fingerprint)
            unique_results.append((doc, score))
    
    return unique_results[:3]  # Return max 3 unique results