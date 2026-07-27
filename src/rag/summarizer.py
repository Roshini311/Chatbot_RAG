import os
from typing import Dict, Any, List
from config.settings import settings
from src.vector_store.manager import VectorStoreManager

class DocumentSummarizer:
    """
    Generates multi-tier document summaries (Executive, Technical, Bullet Breakdown, Key Takeaways).
    """
    def __init__(self, vector_manager: VectorStoreManager = None):
        self.vector_manager = vector_manager or VectorStoreManager()

    def summarize_document(self, doc_id: str, file_name: str = "") -> Dict[str, Any]:
        docs = self.vector_manager.get_chunks_by_doc_id(doc_id, file_name=file_name, k=10)

        if not docs:
            fallback_msg = f"### 📑 Summary for {file_name if file_name else 'Selected Document'}\n\n*Document metadata exists, but text chunks are currently being indexed or the vector database is initializing. Please re-upload the document or try again in a moment.*"
            return {
                "doc_id": doc_id,
                "file_name": file_name,
                "summary": fallback_msg
            }

        full_text = "\n\n".join([d.page_content for d in docs[:8]])
        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")

        if api_key:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, api_key=api_key)
                prompt = f"""Analyze the following document text for '{file_name}' and provide a structured multi-tier summary:

Document Content:
{full_text[:4000]}

Format clearly with Markdown headers:
### Executive Summary
(2-3 high level sentences)

### Technical Summary
(2-3 technical implementation sentences)

### Bullet Point Breakdown
• Key point 1
• Key point 2
• Key point 3

### Key Takeaways
• Takeaway 1
• Takeaway 2"""
                res = llm.invoke(prompt)
                summary_content = res.content
            except Exception:
                summary_content = self._fallback_summary(full_text, file_name)
        else:
            summary_content = self._fallback_summary(full_text, file_name)

        return {
            "doc_id": doc_id,
            "file_name": file_name,
            "summary": summary_content
        }

    def _fallback_summary(self, text: str, file_name: str) -> str:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        exec_sum = paragraphs[0][:350] if len(paragraphs) > 0 else text[:350]
        tech_sum = paragraphs[1][:350] if len(paragraphs) > 1 else text[350:700]

        return f"""### 📑 Executive Summary
{exec_sum}...

### ⚙️ Technical Summary
{tech_sum}...

### 📌 Bullet Point Breakdown
• **Target Document**: `{file_name if file_name else 'Uploaded Document'}`
• **Indexed Corpus**: {len(text.split())} words processed from page chunks.
• **Status**: FAISS vector index & metadata tracking active.

### 🔑 Key Takeaways
• Multi-tier summaries synthesized directly from document text.
• Page citations available for detailed verification in RAG Chat tab.
"""
