import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

FAISS_PATH = "vectorstore"

def get_embeddings_model():
    """
    Returns the HuggingFace embeddings model.
    SentenceTransformers will be downloaded locally automatically if not present.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store():
    """
    Loads the FAISS vector store from disk if it exists.
    Returns None if no vector store exists.
    """
    if os.path.exists(FAISS_PATH):
        embeddings = get_embeddings_model()
        return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def add_documents_to_faiss(documents: list[Document]):
    """
    Adds document chunks to FAISS vector store and saves it locally.
    """
    embeddings = get_embeddings_model()
    
    # Check if vector store already exists
    if os.path.exists(FAISS_PATH):
        vector_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(documents)
    else:
        # Create a new FAISS index from the documents
        vector_store = FAISS.from_documents(documents, embeddings)
        os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
        
    vector_store.save_local(FAISS_PATH)
    return True
