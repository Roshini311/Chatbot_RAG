from src.rag.qa_chain import RAGQuestionAnswering

def test_rag_qa_offline_fallback():
    rag = RAGQuestionAnswering()
    result = rag.answer_question(query="What is artificial intelligence?")
    assert "answer" in result
    assert "citations" in result
    assert "retrieved_context" in result
