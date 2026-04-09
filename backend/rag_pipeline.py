import os
from langchain_openai import ChatOpenAI

def generate_answer(query: str, vector_store) -> dict:
    """
    Retrieves context from the vector store and queries the LLM for an answer.
    """
    # 1. Retrieve the top 4 most relevant chunks
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query)
    
    # 2. Extract and combine the text from chunks
    context = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source_file', 'unknown')}\n{doc.page_content}" for doc in retrieved_docs])
    
    # 3. Create the prompt with context and question
    prompt = f"""You are a helpful AI Knowledge Assistant.
Answer the user's question based ONLY on the provided context below.
If the answer is not contained in the context, politely say "I don't have enough information from the uploaded documents to answer that."
Provide a clear, concise, and professional answer.

Context:
{context}

Question: {query}
Answer:"""

    # 4. Invoke the LLM (OpenAI)
    # Automatically picks up OPENAI_API_KEY from environment variables
    # Defaults to gpt-3.5-turbo which is cheap and fast.
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm.invoke(prompt)
    
    # 5. Format sources for the frontend
    sources = []
    for doc in retrieved_docs:
        sources.append({
            "source": doc.metadata.get("source_file", "Unknown"),
            "content": doc.page_content
        })
        
    return {
        "answer": response.content,
        "sources": sources
    }
