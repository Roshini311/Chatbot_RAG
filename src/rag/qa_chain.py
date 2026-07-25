import os
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
        
        # 1. Retrieve top 4 context chunks
        docs = self.vector_manager.search_hybrid(query, k=4, doc_ids=doc_ids)

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
If the context does not contain sufficient information to answer, state clearly: "I cannot determine the answer from the provided documents."

Conversation History:
{session_history if session_history else 'None'}

Context:
{context_str}

Question: {query}

Provide a clear, precise, and professional answer strictly based on the context above."""

        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")

        if api_key:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, api_key=api_key)
                res = llm.invoke(prompt_text)
                answer = res.content
            except Exception as e:
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
        summary_snippets = [f"• From '{d.metadata.get('file_name')} (Page {d.metadata.get('page_number')})': {d.page_content[:250]}..." for d in docs]
        return f"*(Note: Running in offline context-extraction mode. Set OPENAI_API_KEY in .env for full GPT synthesis.)*\n\nBased on your documents, the top retrieved sections addressing '{query}' are:\n\n" + "\n\n".join(summary_snippets)
