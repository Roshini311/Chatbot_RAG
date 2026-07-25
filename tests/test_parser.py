from src.document_processing.chunker import DocumentChunker

def test_chunker_basic():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    pages_data = [
        {
            "doc_id": "test_doc_1",
            "file_name": "sample.pdf",
            "page_number": 1,
            "text": "This is a test paragraph designed to verify that the chunking mechanism correctly splits long texts while retaining exact page metadata."
        }
    ]

    chunks = chunker.create_chunks(pages_data)
    assert len(chunks) > 0
    assert chunks[0]["doc_id"] == "test_doc_1"
    assert chunks[0]["page_number"] == 1
    assert "file_name" in chunks[0]
