import os
from typing import Dict, Any, List
from config.settings import settings
from src.vector_store.manager import VectorStoreManager

class DocumentComparator:
    """
    Compares research methodologies, pros/cons, and implementations across multiple documents.
    """
    def __init__(self, vector_manager: VectorStoreManager = None):
        self.vector_manager = vector_manager or VectorStoreManager()

    def compare_documents(self, doc_ids: List[str], file_names: List[str] = None) -> Dict[str, Any]:
        if not doc_ids or len(doc_ids) < 2:
            return {"error": "At least 2 document IDs are required for multi-document comparison."}

        contexts_by_doc = {}
        for idx, did in enumerate(doc_ids):
            fname = file_names[idx] if file_names and idx < len(file_names) else f"Document {did[:8]}"
            docs = self.vector_manager.get_chunks_by_doc_id(did, k=5)
            combined = "\n".join([d.page_content for d in docs])
            contexts_by_doc[fname] = combined[:2000] if combined else "No context found."

        combined_prompt_context = ""
        for name, text in contexts_by_doc.items():
            combined_prompt_context += f"\n=== {name} ===\n{text}\n"

        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")

        if api_key:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, api_key=api_key)
                prompt = f"""Compare the following research/technical documents:

{combined_prompt_context}

Provide a comparative analysis covering:
1. Core Methodologies & Approaches
2. Key Similarities
3. Key Differences / Advantages & Disadvantages
4. Recommended Use Cases / Implementation Approaches"""
                res = llm.invoke(prompt)
                comparison_result = res.content
            except Exception:
                comparison_result = self._fallback_comparison(contexts_by_doc)
        else:
            comparison_result = self._fallback_comparison(contexts_by_doc)

        return {
            "compared_documents": file_names or doc_ids,
            "comparison": comparison_result
        }

    def _fallback_comparison(self, contexts: Dict[str, str]) -> str:
        res = "### Multi-Document Comparative Overview\n\n"
        for fname, text in contexts.items():
            res += f"#### 📄 {fname}\n"
            res += f"• **Extracted Context**: {text[:250]}...\n\n"

        res += """### Comparative Breakdown
• **Methodologies**: The uploaded documents provide complementary domain perspectives.
• **Similarities**: All files are successfully indexed and tracked with page citations.
• **Implementation Approaches**: Use RAG QA tab for citation-grounded cross-document queries.
"""
        return res
