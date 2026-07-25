from typing import List, Dict, Any
from config.settings import settings

class DocumentChunker:
    """
    Splits page text into overlapping chunks while preserving page & document metadata.
    """
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def create_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        chunk_counter = 0

        for page in pages_data:
            text = page["text"]
            doc_id = page["doc_id"]
            file_name = page["file_name"]
            page_number = page["page_number"]

            if len(text) <= self.chunk_size:
                chunks.append({
                    "chunk_id": f"{doc_id}_c{chunk_counter}",
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "page_number": page_number,
                    "text": text
                })
                chunk_counter += 1
            else:
                start = 0
                while start < len(text):
                    end = start + self.chunk_size
                    chunk_text = text[start:end]

                    chunks.append({
                        "chunk_id": f"{doc_id}_c{chunk_counter}",
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "page_number": page_number,
                        "text": chunk_text
                    })
                    chunk_counter += 1
                    start += (self.chunk_size - self.chunk_overlap)

        return chunks
