import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

def process_document(file_path: str, file_name: str) -> List[Document]:
    """
    Extracts text from a document and splits it into chunks.
    Raises ValueError if the file format is unsupported.
    """
    _, ext = os.path.splitext(file_name.lower())
    
    # 1. Load the document
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
        
    documents = loader.load()
    
    # Add the source filename to metadata for citation purposes
    for doc in documents:
        doc.metadata["source_file"] = file_name
    
    # 2. Chunk the text
    # 500-1000 tokens is good; characters ~ 1000 with 200 overlap is standard.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks
