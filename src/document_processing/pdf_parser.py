import os
from typing import List, Dict, Any

class PDFParser:
    """
    Parses PDF files page-by-page preserving structural page metadata.
    Uses PyMuPDF (fitz) with a fallback to pypdf.
    """
    def __init__(self):
        pass

    def extract_text_with_metadata(self, pdf_path: str, doc_id: str, file_name: str) -> List[Dict[str, Any]]:
        extracted_pages = []
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()
                if text:
                    extracted_pages.append({
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "page_number": page_num + 1,
                        "text": text
                    })
            doc.close()
        except Exception:
            # Fallback to pypdf if PyMuPDF fails or isn't available
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    extracted_pages.append({
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "page_number": page_num + 1,
                        "text": text
                    })

        return extracted_pages
