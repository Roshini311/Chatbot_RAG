import os
from typing import List, Dict, Any
from config.settings import settings

class FallbackEmbeddings:
    """Lightweight 384-dimensional embedding generator fallback."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float((hash(t + str(i)) % 100)) / 1000.0 for i in range(384)] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float((hash(text + str(i)) % 100)) / 1000.0 for i in range(384)]

class VectorStoreManager:
    """
    Manages dense vector index & metadata persistence using FAISS and HuggingFace Embeddings.
    """
    def __init__(self, vector_db_dir: str = None):
        self.vector_db_dir = vector_db_dir or settings.VECTOR_DB_DIR
        self.embeddings = self._get_embeddings()
        self.vector_store = self._load_vector_store()

    def _get_embeddings(self):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                try:
                    from langchain_openai import OpenAIEmbeddings
                    return OpenAIEmbeddings()
                except Exception:
                    print("Notice: Using lightweight FallbackEmbeddings engine.")
                    return FallbackEmbeddings()

    def _load_vector_store(self):
        try:
            from langchain_community.vectorstores import FAISS
            if os.path.exists(self.vector_db_dir) and os.path.exists(os.path.join(self.vector_db_dir, "index.faiss")):
                return FAISS.load_local(
                    self.vector_db_dir, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
        except Exception as e:
            print(f"Notice loading FAISS index: {e}")
        return None

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        if not chunks:
            return False

        try:
            from langchain_core.documents import Document
            from langchain_community.vectorstores import FAISS

            documents = []
            for c in chunks:
                doc = Document(
                    page_content=c["text"],
                    metadata={
                        "doc_id": c["doc_id"],
                        "file_name": c["file_name"],
                        "page_number": c["page_number"],
                        "chunk_id": c.get("chunk_id", "")
                    }
                )
                documents.append(doc)

            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)

            os.makedirs(self.vector_db_dir, exist_ok=True)
            self.vector_store.save_local(self.vector_db_dir)
            return True
        except Exception as e:
            print(f"Error adding chunks to FAISS: {e}")
            return False

    def search_similarity(self, query: str, k: int = 4, doc_ids: List[str] = None) -> List[Any]:
        if self.vector_store is None:
            return []

        try:
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k*3 if doc_ids else k)
            
            filtered = []
            for doc, score in docs_and_scores:
                if doc_ids:
                    if doc.metadata.get("doc_id") in doc_ids:
                        filtered.append(doc)
                else:
                    filtered.append(doc)

            return filtered[:k]
        except Exception as e:
            print(f"Error executing similarity search: {e}")
            return []

    def search_hybrid(self, query: str, k: int = 4, doc_ids: List[str] = None) -> List[Any]:
        dense_docs = self.search_similarity(query, k=k*2, doc_ids=doc_ids)
        query_words = set(query.lower().split())

        def keyword_score(doc):
            content_words = set(doc.page_content.lower().split())
            overlap = len(query_words.intersection(content_words))
            return overlap

        ranked = sorted(dense_docs, key=lambda d: keyword_score(d), reverse=True)
        return ranked[:k]
