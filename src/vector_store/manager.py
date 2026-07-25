import os
import math
import re
from typing import List, Dict, Any
from config.settings import settings

class UltraFastEmbeddings:
    """
    Ultra-fast 384-dimensional dense feature hashing vectorizer.
    Runs in <1ms, consumes <2MB RAM, requires 0 downloads, zero OOM risks.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _text_to_vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        words = re.findall(r'\w+', text.lower())
        if not words:
            return vec

        for i, word in enumerate(words):
            # Unigram hash
            h1 = abs(hash(word)) % self.dim
            vec[h1] += 1.0

            # Bigram hash for context
            if i < len(words) - 1:
                bigram = word + "_" + words[i+1]
                h2 = abs(hash(bigram)) % self.dim
                vec[h2] += 1.5

        # L2 Normalization
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._text_to_vector(text)

class VectorStoreManager:
    """
    High-performance Vector Store Manager backed by FAISS and UltraFastEmbeddings.
    """
    def __init__(self, vector_db_dir: str = None):
        self.vector_db_dir = vector_db_dir or settings.VECTOR_DB_DIR
        self.embeddings = UltraFastEmbeddings()
        self.vector_store = self._load_vector_store()

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
            print(f"FAISS index notice: {e}")
        return None

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        if not chunks:
            return False

        try:
            from langchain_core.documents import Document
            from langchain_community.vectorstores import FAISS

            documents = [
                Document(
                    page_content=c["text"],
                    metadata={
                        "doc_id": c["doc_id"],
                        "file_name": c["file_name"],
                        "page_number": c["page_number"],
                        "chunk_id": c.get("chunk_id", "")
                    }
                )
                for c in chunks
            ]

            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)

            os.makedirs(self.vector_db_dir, exist_ok=True)
            self.vector_store.save_local(self.vector_db_dir)
            return True
        except Exception as e:
            print(f"Error indexing chunks in FAISS: {e}")
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
            print(f"Error performing similarity search: {e}")
            return []

    def search_hybrid(self, query: str, k: int = 4, doc_ids: List[str] = None) -> List[Any]:
        dense_docs = self.search_similarity(query, k=k*2, doc_ids=doc_ids)
        query_words = set(re.findall(r'\w+', query.lower()))

        def keyword_score(doc):
            content_words = set(re.findall(r'\w+', doc.page_content.lower()))
            overlap = len(query_words.intersection(content_words))
            return overlap

        ranked = sorted(dense_docs, key=lambda d: keyword_score(d), reverse=True)
        return ranked[:k]
