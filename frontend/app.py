import streamlit as st
import requests
import os

# Set page configuration for a premium look
st.set_page_config(
    page_title="AI Knowledge Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI enhancements
st.markdown("""
<style>
    .stApp {
        background-color: #0f111a;
        color: #e0e0e0;
    }
    .stChatInputContainer {
        border-radius: 10px;
    }
    h1 {
        color: #4df0e1;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #4df0e1;
        color: #0f111a;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #26c6da;
        box-shadow: 0px 4px 15px rgba(38, 198, 218, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Knowledge Assistant")
st.markdown("Upload your documents (PDF, TXT, DOCX) in the sidebar and ask questions about them!")

# Backend URL - Use local container communication for Docker, or external if specified
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Sidebar for document upload
with st.sidebar:
    st.header("📂 1. Upload Documents")
    uploaded_file = st.file_uploader("Choose a file to index", type=["pdf", "txt", "docx"])
    
    if st.button("Process & Embed Document"):
        if uploaded_file is not None:
            with st.spinner("Extracting and embedding text..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error(f"Backend Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Connection Error: Is the FastAPI backend running on port 8000?\n\n{e}")
        else:
            st.warning("Please upload a file above first.")
            
    st.markdown("---")
    st.header("⚙️ 2. Status")
    st.info("Make sure the backend API is actively running using `uvicorn backend.main:app`.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Please upload a document in the sidebar to get started, then ask me anything."}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if they exist for assistant replies
        if message.get("role") == "assistant" and message.get("sources"):
            with st.expander("🔍 View Retrieved Context / Sources"):
                for idx, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {idx+1}: {source['source']}**")
                    st.caption(f"_{source['content']}_")

# React to user input
if prompt := st.chat_input("Ask a question based on uploaded documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant Response Placeholder
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document database..."):
            try:
                response = requests.post(f"{BACKEND_URL}/query", json={"query": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "I could not generate an answer.")
                    sources = data.get("sources", [])
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Display sources in an expander for transparency
                    if sources:
                        with st.expander("🔍 View Retrieved Context / Sources"):
                            for idx, source in enumerate(sources):
                                st.markdown(f"**Source {idx+1}: {source['source']}**")
                                st.caption(f"_{source['content']}_")
                                
                    # Save assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_msg = response.json().get('detail', 'Unknown error')
                    st.error(f"Error: {error_msg}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}. Ensure API is running on localhost:8000.")
