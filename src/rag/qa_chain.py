import os
import re
from typing import Dict, Any, List
from config.settings import settings
from src.vector_store.manager import VectorStoreManager

class RAGQuestionAnswering:
    """
    RAG QA Chain with citation tracking and conversation memory.
    """
    def __init__(self, vector_manager: VectorStoreManager = None):
        self.vector_manager = vector_manager or VectorStoreManager()

    def answer_question(
        self, 
        query: str, 
        session_history: str = "", 
        doc_ids: List[str] = None
    ) -> Dict[str, Any]:
        
        # Check if vector store is initialized or empty
        if self.vector_manager.vector_store is None:
            return {
                "answer": "⚠️ **No documents have been uploaded yet.** Please go to the **'📄 Document Management'** tab to upload your PDF files first!",
                "citations": [],
                "retrieved_context": []
            }

        # 1. Retrieve top 6 context chunks
        docs = self.vector_manager.search_hybrid(query, k=6, doc_ids=doc_ids)

        # Fallback for broad/comparative queries if initial search yields few results
        broad_keywords = ["common", "both", "compare", "difference", "all", "summary", "overview", "relationship"]
        if not docs or any(k in query.lower() for k in broad_keywords):
            broad_docs = self.vector_manager.search_similarity("overview background summary key findings technical details", k=6, doc_ids=doc_ids)
            if broad_docs:
                # Merge unique docs
                existing_contents = {d.page_content for d in docs}
                for bd in broad_docs:
                    if bd.page_content not in existing_contents:
                        docs.append(bd)
                        existing_contents.add(bd.page_content)

        if not docs:
            return {
                "answer": "I cannot determine the answer because no matching document context was found in the database.",
                "citations": [],
                "retrieved_context": []
            }

        # 2. Build context string with exact page and document citations
        context_str = ""
        citations = []
        seen_citations = set()

        for d in docs:
            doc_name = d.metadata.get("file_name", "Unknown Document")
            page_no = d.metadata.get("page_number", "N/A")
            context_str += f"\n--- Source: {doc_name} (Page {page_no}) ---\n{d.page_content}\n"
            
            cit_key = f"{doc_name}_p{page_no}"
            if cit_key not in seen_citations:
                seen_citations.add(cit_key)
                citations.append({"document": doc_name, "page": page_no})

        # 3. Prompt structuring
        prompt_text = f"""You are an AI Research & Knowledge Assistant. Answer the user's question using ONLY the provided document context below.
If comparing multiple documents or identifying common features, analyze the similarities and differences present in the context.
If the context does not contain sufficient information to answer, state clearly: "I cannot determine the answer from the provided documents."

Conversation History:
{session_history if session_history else 'None'}

Context:
{context_str}

Question: {query}

Provide a clear, precise, and structured answer strictly based on the context above."""

        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")

        if api_key:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, api_key=api_key)
                res = llm.invoke(prompt_text)
                answer = res.content
            except Exception:
                answer = self._generate_fallback_answer(query, docs)
        else:
            answer = self._generate_fallback_answer(query, docs)

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_context": [d.page_content for d in docs]
        }

    def _generate_fallback_answer(self, query: str, docs: List[Any]) -> str:
        """
        Extracted context summary when OpenAI API Key is absent or unreachable.
        """
        summary_snippets = [f"• **Source: {d.metadata.get('file_name')} (Page {d.metadata.get('page_number')})**:\n{d.page_content[:300]}..." for d in docs]
        return f"*(Note: Running in offline context-extraction mode. Set OPENAI_API_KEY in .env for full GPT synthesis.)*\n\nBased on your documents, the relevant context sections addressing **'{query}'** are:\n\n" + "\n\n".join(summary_snippets)
