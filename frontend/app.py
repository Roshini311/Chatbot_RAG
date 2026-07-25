import os
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="AI Research & Knowledge Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    h1, h2, h3 { color: #58a6ff; font-family: 'Inter', sans-serif; }
    .stButton>button {
        background-color: #238636;
        color: #ffffff;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover { background-color: #2ea043; }
    .metric-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .citation-badge {
        background-color: #1f6feb;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.title("🔬 AI Research & Knowledge Assistant")
st.caption("Enterprise RAG Engine • TensorFlow ML Classifier • Page-Level Citations • Comparative Analytics")

# Sidebar Status & API Config
with st.sidebar:
    st.header("⚙️ System Status")
    try:
        res = requests.get(f"{BACKEND_URL}/", timeout=3)
        if res.status_code == 200:
            st.success("API Backend: Connected 🟢")
        else:
            st.warning("API Backend: Warning 🟡")
    except Exception:
        st.error("API Backend: Offline 🔴\n\nRun `python main.py` to start backend.")

    st.markdown("---")
    st.markdown("### 🛠️ Quick Navigation")
    st.info("Use the tabs on the main screen to navigate between Document Ingestion, RAG Q&A, Document Comparison, and Analytics.")

# Tabs Setup
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Document Management", 
    "🔍 Semantic Search & RAG Chat", 
    "📊 Summarize & Compare", 
    "📈 System Analytics"
])

# ---------------------------------------------------------
# TAB 1: Document Management & Auto-Classification
# ---------------------------------------------------------
with tab1:
    st.header("📄 Upload & Ingest PDF Documents")
    st.markdown("Uploaded PDFs will be parsed page-by-page, categorized using a **TensorFlow Deep Learning Model**, chunked, and indexed into the vector store.")

    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])
    if uploaded_file is not None:
        if st.button("🚀 Process & Auto-Categorize Document"):
            with st.spinner("Extracting pages, categorizing, and indexing chunks..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = requests.post(f"{BACKEND_URL}/documents/upload", files=files, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Successfully processed '{data['file_name']}'!")
                        st.json(data)
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

    st.markdown("---")
    st.subheader("📚 Ingested Document Metadata Registry")
    
    try:
        list_resp = requests.get(f"{BACKEND_URL}/documents/list", timeout=5)
        if list_resp.status_code == 200:
            docs = list_resp.json()
            if docs:
                for doc in docs:
                    with st.expander(f"📄 {doc['file_name']} — Category: {doc['category']}"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Doc ID", doc['doc_id'][:8])
                        col2.metric("Total Pages", doc['total_pages'])
                        col3.metric("Total Chunks", doc['total_chunks'])
                        col4.metric("Status", doc['processing_status'])

                        if st.button(f"Delete {doc['file_name']}", key=f"del_{doc['doc_id']}"):
                            del_resp = requests.delete(f"{BACKEND_URL}/documents/{doc['doc_id']}", timeout=5)
                            if del_resp.status_code == 200:
                                st.success(f"Deleted {doc['file_name']}")
                                st.rerun()
            else:
                st.info("No documents uploaded yet. Upload a PDF above to begin.")
    except Exception as e:
        st.warning(f"Could not load document registry: {e}")

# ---------------------------------------------------------
# TAB 2: Semantic Search & RAG Chat
# ---------------------------------------------------------
with tab2:
    st.header("🔍 RAG Research Assistant with Citations")
    st.markdown("Ask complex questions across your document repository. All answers are strictly grounded with **page & file citations**.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message and message["citations"]:
                st.markdown("**📌 Sources & Page Citations:**")
                c_html = ""
                for c in message["citations"]:
                    c_html += f"<span class='citation-badge'>📄 {c['document']} (Page {c['page']})</span> "
                st.markdown(c_html, unsafe_allow_html=True)

    # Chat Input
    if query := st.chat_input("Ask a research or technical question..."):
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Searching vector index & synthesizing context-grounded answer..."):
                try:
                    payload = {"query": query, "session_history": ""}
                    resp = requests.post(f"{BACKEND_URL}/search/qa", json=payload, timeout=30)
                    if resp.status_code == 200:
                        res_data = resp.json()
                        answer = res_data["answer"]
                        citations = res_data.get("citations", [])

                        st.markdown(answer)

                        if citations:
                            st.markdown("**📌 Sources & Page Citations:**")
                            c_html = ""
                            for c in citations:
                                c_html += f"<span class='citation-badge'>📄 {c['document']} (Page {c['page']})</span> "
                            st.markdown(c_html, unsafe_allow_html=True)

                        with st.expander("🔍 View Raw Retrieved Context Chunks"):
                            for idx, ctx in enumerate(res_data.get("retrieved_context", [])):
                                st.markdown(f"**Chunk {idx+1}:**")
                                st.caption(ctx)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations
                        })
                    else:
                        st.error(f"QA Error: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error querying backend: {e}")

# ---------------------------------------------------------
# TAB 3: Summarize & Compare
# ---------------------------------------------------------
with tab3:
    st.header("📊 Multi-Document Comparison & Summarization")

    try:
        list_resp = requests.get(f"{BACKEND_URL}/documents/list", timeout=5)
        docs = list_resp.json() if list_resp.status_code == 200 else []
    except Exception:
        docs = []

    doc_options = {d["doc_id"]: d["file_name"] for d in docs} if docs else {}

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📑 Document Summarizer")
        selected_doc_id = st.selectbox("Select document to summarize", options=list(doc_options.keys()), format_func=lambda x: doc_options[x] if x in doc_options else x)
        
        if st.button("Generate Summary") and selected_doc_id:
            with st.spinner("Generating multi-tier summary..."):
                resp = requests.post(f"{BACKEND_URL}/analysis/summarize", json={"doc_id": selected_doc_id, "file_name": doc_options.get(selected_doc_id, "")}, timeout=30)
                if resp.status_code == 200:
                    sum_data = resp.json()
                    st.markdown(sum_data.get("summary", "No summary content generated."))
                else:
                    st.error("Summarization failed.")

    with col_b:
        st.subheader("⚖️ Multi-Document Comparator")
        selected_compare_ids = st.multiselect("Select 2+ documents to compare", options=list(doc_options.keys()), format_func=lambda x: doc_options[x] if x in doc_options else x)
        
        if st.button("Compare Selected Documents") and len(selected_compare_ids) >= 2:
            with st.spinner("Analyzing comparative methodologies & pros/cons..."):
                fnames = [doc_options[did] for did in selected_compare_ids]
                resp = requests.post(f"{BACKEND_URL}/analysis/compare", json={"doc_ids": selected_compare_ids, "file_names": fnames}, timeout=30)
                if resp.status_code == 200:
                    cmp_data = resp.json()
                    st.markdown(cmp_data.get("comparison", "No comparison generated."))
                else:
                    st.error("Comparison failed.")

# ---------------------------------------------------------
# TAB 4: System Analytics
# ---------------------------------------------------------
with tab4:
    st.header("📈 System Analytics & Usage Metrics")
    st.markdown("Track index size, processed pages, query history, and **TensorFlow domain classification statistics**.")

    if st.button("🔄 Refresh Metrics"):
        st.rerun()

    try:
        stats_resp = requests.get(f"{BACKEND_URL}/analytics/stats", timeout=5)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Indexed Documents", stats["total_documents"])
            c2.metric("Total Extracted Pages", stats["total_pages"])
            c3.metric("Total Vector Chunks", stats["total_chunks"])
            c4.metric("Total Queries Executed", stats["total_queries"])

            st.markdown("---")
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("🤖 TensorFlow Category Distribution")
                cat_data = stats.get("category_distribution", {})
                if cat_data:
                    st.bar_chart(cat_data)
                else:
                    st.info("No category data available yet.")

            with col_right:
                st.subheader("⚙️ Processing Status Breakdown")
                st_data = stats.get("status_distribution", {})
                if st_data:
                    st.bar_chart(st_data)
                else:
                    st.info("No status data available yet.")
        else:
            st.error("Could not fetch analytics statistics.")
    except Exception as e:
        st.error(f"Error connecting to analytics engine: {e}")
