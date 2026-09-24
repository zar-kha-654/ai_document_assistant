import os
import re
import tempfile
import numpy as np
import streamlit as st
import faiss
import gdown
from pypdf import PdfReader
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
from groq import Groq

# ------------------------------------------------------------------------------
# Page Config & Custom Modern CSS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuMind AI | Document Intelligence Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Global Theme & Font Adjustments */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem !important;
    }
    .sub-title {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    
    /* Modern Card Layouts */
    .custom-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Upload Drag and Drop zone customization */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #3B82F6 !important;
        background-color: #111827 !important;
        border-radius: 12px !important;
    }
    
    /* Badge tags */
    .tag-badge {
        display: inline-block;
        background-color: #1E3A8A;
        color: #93C5FD;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        margin-right: 6px;
    }
    
    .chunk-card {
        background-color: #0F172A;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
        font-family: 'Fira Code', monospace;
        font-size: 0.88rem;
        color: #E2E8F0;
    }

    /* Primary Buttons */
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 1. Caching Heavy Resources & Client Setup (UNTOUCHED LOGIC)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Embedding Model...")
def load_embedding_model():
    """Loads and caches the SentenceTransformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_groq_client():
    """Retrieves Groq client using secrets or environment variable."""
    api_key = None
    
    # Check Streamlit secrets
    try:
        if "groq_api_key" in st.secrets:
            api_key = st.secrets["groq_api_key"]
    except Exception:
        pass
    
    # Fallback to environment variable
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("🔑 Groq API key not found! Please configure `groq_api_key` in `.streamlit/secrets.toml`.")
        st.stop()
        
    return Groq(api_key=api_key)


# ------------------------------------------------------------------------------
# 2. Text Extraction Functions (UNTOUCHED LOGIC)
# ------------------------------------------------------------------------------
def clean_text(text):
    """Normalizes whitespace and removes unprintable characters."""
    if not text:
        return ""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(file_path, file_name):
    """Extracts text page-by-page from a PDF document."""
    documents = []
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            cleaned = clean_text(raw_text)
            if cleaned:
                documents.append({
                    "file_name": file_name,
                    "page": i + 1,
                    "text": cleaned
                })
    except Exception as e:
        st.warning(f"Failed to read PDF '{file_name}': {e}")
    return documents


def extract_text_from_docx(file_path, file_name):
    """Extracts text from a DOCX document."""
    try:
        doc = DocxDocument(file_path)
        text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        cleaned = clean_text(text)
        if cleaned:
            return [{
                "file_name": file_name,
                "page": None,
                "text": cleaned
            }]
    except Exception as e:
        st.warning(f"Failed to read DOCX '{file_name}': {e}")
    return []


def extract_text_from_txt(file_path, file_name):
    """Extracts text from a plain TXT or Markdown file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        cleaned = clean_text(text)
        if cleaned:
            return [{
                "file_name": file_name,
                "page": None,
                "text": cleaned
            }]
    except Exception as e:
        st.warning(f"Failed to read TXT/MD '{file_name}': {e}")
    return []


def process_file_path(file_path, file_name):
    """Routes file extraction based on extension."""
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path, file_name)
    elif ext == ".docx":
        return extract_text_from_docx(file_path, file_name)
    elif ext in [".txt", ".md"]:
        return extract_text_from_txt(file_path, file_name)
    else:
        st.warning(f"Unsupported file format or missing extension for: {file_name}")
    return []


# ------------------------------------------------------------------------------
# 3. Google Drive Handling (UNTOUCHED LOGIC)
# ------------------------------------------------------------------------------
def fetch_from_google_drive(drive_url):
    """Downloads files/folders from Google Drive using gdown."""
    downloaded_docs = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            if "folders/" in drive_url or "folder" in drive_url:
                folder_files = gdown.download_folder(url=drive_url, output=tmp_dir, quiet=True)
                if folder_files:
                    for fpath in folder_files:
                        if os.path.isfile(fpath):
                            fname = os.path.basename(fpath)
                            extracted = process_file_path(fpath, fname)
                            downloaded_docs.extend(extracted)
            else:
                fpath = gdown.download(url=drive_url, output=tmp_dir + "/", quiet=True)
                
                if fpath and os.path.exists(fpath):
                    fname = os.path.basename(fpath)
                    
                    if not os.path.splitext(fname)[1]:
                        for known_ext in [".pdf", ".docx", ".txt", ".md"]:
                            if known_ext in drive_url.lower():
                                new_path = fpath + known_ext
                                os.rename(fpath, new_path)
                                fpath = new_path
                                fname = fname + known_ext
                                break
                    
                    extracted = process_file_path(fpath, fname)
                    downloaded_docs.extend(extracted)
                else:
                    st.error("Google Drive download failed. Ensure the link is set to 'Anyone with the link'.")
        except Exception as e:
            st.error(f"Error fetching from Google Drive: {e}")
            
    return downloaded_docs


# ------------------------------------------------------------------------------
# 4. Paragraph-Aware Chunking (UNTOUCHED LOGIC)
# ------------------------------------------------------------------------------
def chunk_text(documents, target_chunk_size=1000, overlap_size=200):
    """
    Splits document text cleanly by paragraphs first to preserve full sentences.
    Falls back to character slicing only when a single paragraph exceeds target size.
    """
    chunks = []
    
    for doc in documents:
        text = doc["text"]
        paragraphs = text.split("\n\n")
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) <= target_chunk_size:
                current_chunk += ("\n\n" + para) if current_chunk else para
            else:
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "file_name": doc["file_name"],
                        "page": doc["page"]
                    })
                
                if len(para) > target_chunk_size:
                    start = 0
                    while start < len(para):
                        end = start + target_chunk_size
                        sub_str = para[start:end]
                        chunks.append({
                            "text": sub_str.strip(),
                            "file_name": doc["file_name"],
                            "page": doc["page"]
                        })
                        start += (target_chunk_size - overlap_size)
                    current_chunk = ""
                else:
                    current_chunk = para

        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "file_name": doc["file_name"],
                "page": doc["page"]
            })

    return chunks


# ------------------------------------------------------------------------------
# 5. Hybrid Search Pipeline (UNTOUCHED LOGIC)
# ------------------------------------------------------------------------------
def build_vector_store(chunks, embed_model):
    """Generates embeddings and initializes FAISS vector index."""
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return index, embeddings


def keyword_search_score(query, text):
    """Calculates keyword match score based on query terms."""
    keywords = set(re.findall(r"\w+", query.lower()))
    if not keywords:
        return 0.0
    
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw in text_lower)
    return matches / len(keywords)


def hybrid_search(query, chunks, index, embed_model, top_k=5, alpha=0.8):
    """Combines semantic (FAISS) and keyword search scores."""
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    
    num_candidates = min(len(chunks), max(top_k * 3, 10))
    distances, indices = index.search(query_emb, num_candidates)
    
    results = []
    for idx, sem_score in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        
        kw_score = keyword_search_score(query, chunk["text"])
        combined_score = alpha * float(sem_score) + (1 - alpha) * float(kw_score)
        
        results.append({
            "chunk": chunk,
            "score": combined_score,
            "sem_score": float(sem_score),
            "kw_score": float(kw_score)
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ------------------------------------------------------------------------------
# 6. Streamlit Modern UI Layout & State Management
# ------------------------------------------------------------------------------
def main():
    # Session State Initialization
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "docs_processed" not in st.session_state:
        st.session_state.docs_processed = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "ratings" not in st.session_state:
        st.session_state.ratings = {}

    embed_model = load_embedding_model()

    # ---------------- Sidebar: Knowledge Base Management ----------------
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
        st.title("Knowledge Base")
        st.caption("Upload documents to build your vector search index.")
        st.markdown("---")
        
        st.subheader("1. Local Files")
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        st.markdown("""
            <div style='margin-top: -10px; margin-bottom: 15px;'>
                <span class='tag-badge'>PDF</span>
                <span class='tag-badge'>DOCX</span>
                <span class='tag-badge'>TXT</span>
                <span class='tag-badge'>MD</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.subheader("2. Cloud Source")
        drive_url = st.text_input("Google Drive Folder or File Link", placeholder="https://drive.google.com/...")
        
        st.markdown("<br>", unsafe_allow_html=True)
        process_btn = st.button("🚀 Process & Build Index", type="primary", use_container_width=True)

        st.markdown("---")
        st.subheader("📊 System Status")
        if st.session_state.chunks:
            st.success(f"🟢 **Index Ready**\n\n**{len(st.session_state.chunks)}** chunks active in memory.")
        else:
            st.info("🟡 **Index Empty**\n\nPlease upload documents to begin.")

    # Processing Action
    if process_btn:
        all_extracted_docs = []
        processed_file_names = []

        if uploaded_files:
            for ufile in uploaded_files:
                ext = os.path.splitext(ufile.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(ufile.getvalue())
                    tmp_path = tmp.name
                
                docs = process_file_path(tmp_path, ufile.name)
                all_extracted_docs.extend(docs)
                processed_file_names.append(ufile.name)
                os.remove(tmp_path)

        if drive_url.strip():
            with st.spinner("Fetching from Google Drive..."):
                drive_docs = fetch_from_google_drive(drive_url.strip())
                all_extracted_docs.extend(drive_docs)
                processed_file_names.append("Google Drive Source")

        if all_extracted_docs:
            with st.spinner("Processing & embedding chunks..."):
                chunks = chunk_text(all_extracted_docs)
                faiss_idx, _ = build_vector_store(chunks, embed_model)
                
                st.session_state.chunks = chunks
                st.session_state.faiss_index = faiss_idx
                st.session_state.docs_processed = processed_file_names
                
            st.toast(f"Success! Indexed {len(chunks)} text chunks.", icon="🎉")
        else:
            st.error("No text could be extracted. Check your uploaded files or link.")

    # ---------------- Main Dashboard Header ----------------
    st.markdown('<div class="main-title">DocuMind AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Modern Hybrid RAG Assistant with Paragraph Chunk Inspection</div>', unsafe_allow_html=True)

    # Tabs for Workspace & Chunk Analysis
    tab_chat, tab_chunks = st.tabs(["💬 AI Chat Assistant", "🔍 Chunk Inspector & Data Breakdown"])

    # ---------------- TAB 1: Chat & QA ----------------
    with tab_chat:
        # Display existing message history
        for idx, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Render sources card for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Retrieved Sources & Relevance Scores"):
                        for s_idx, res in enumerate(message["sources"], 1):
                            chunk = res["chunk"]
                            page_str = f" | Page {chunk['page']}" if chunk['page'] is not None else ""
                            st.markdown(f"**Source {s_idx}:** `{chunk['file_name']}`{page_str} (Score: `{res['score']:.3f}`)")
                            st.caption(chunk["text"])
                            st.divider()

                # Render Rating/Review Component
                if message["role"] == "assistant":
                    rating_key = f"rating_{idx}"
                    st.caption("How was this response?")
                    col_rate1, col_rate2 = st.columns([1, 4])
                    with col_rate1:
                        user_rating = st.feedback("thumbs", key=f"fb_{idx}")
                    with col_rate2:
                        feedback_text = st.text_input("Optional feedback", key=f"txt_{idx}", label_visibility="collapsed", placeholder="Add feedback details...")
                    if user_rating is not None:
                        st.session_state.ratings[idx] = {"score": user_rating, "feedback": feedback_text}

        # Chat Input Bar
        user_query = st.chat_input("Ask anything about your documents...")

        if user_query:
            if not st.session_state.chunks or st.session_state.faiss_index is None:
                st.warning("⚠️ Please upload and process documents in the sidebar first.")
                return

            # Append user prompt
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Perform Hybrid Search
            relevant_results = hybrid_search(
                query=user_query,
                chunks=st.session_state.chunks,
                index=st.session_state.faiss_index,
                embed_model=embed_model,
                top_k=5
            )

            context_parts = []
            for r in relevant_results:
                c = r["chunk"]
                page_info = f" (Page {c['page']})" if c['page'] is not None else ""
                context_parts.append(f"[Source: {c['file_name']}{page_info}]\n{c['text']}")
            
            formatted_context = "\n\n====================\n\n".join(context_parts)

            system_prompt = (
                "You are a helpful assistant. Answer the question based on the provided document excerpts below.\n"
                "Use the provided context to answer as thoroughly as possible.\n"
                "If the context truly lacks the facts needed to answer, reply with:\n"
                "'I'm sorry, but the provided document context does not contain this information.'"
            )
            
            user_prompt = f"DOCUMENT EXCERPTS:\n{formatted_context}\n\nQUESTION: {user_query}"

            # Query Groq Engine
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document context..."):
                    client = get_groq_client()
                    
                    candidate_models = []
                    try:
                        fetched_models = client.models.list().data
                        for m in fetched_models:
                            model_id = m.id.lower()
                            if not any(x in model_id for x in ["whisper", "guard", "eval", "tool", "embedding"]):
                                candidate_models.append(m.id)
                    except Exception:
                        pass

                    if not candidate_models:
                        candidate_models = [
                            "llama-3.3-70b-versatile",
                            "llama-3.1-8b-instant",
                            "llama-3.2-3b-preview",
                            "llama3-70b-8192"
                        ]

                    answer = None
                    error_details = ""

                    for model_name in candidate_models:
                        try:
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                temperature=0.1
                            )
                            answer = response.choices[0].message.content
                            break
                        except Exception as err:
                            error_details += f"\n- {model_name}: {err}"
                            continue

                    if answer:
                        st.markdown(answer)
                        
                        with st.expander("📚 View Retrieved Sources & Relevance Scores"):
                            for s_idx, res in enumerate(relevant_results, 1):
                                chunk = res["chunk"]
                                page_str = f" | Page {chunk['page']}" if chunk['page'] is not None else ""
                                st.markdown(f"**Source {s_idx}:** `{chunk['file_name']}`{page_str} (Score: `{res['score']:.3f}`)")
                                st.caption(chunk["text"])
                                st.divider()
                        
                        # Store in history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": relevant_results
                        })
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate a response from Groq.")
                        with st.expander("View Error Details"):
                            st.code(error_details)

    # ---------------- TAB 2: Chunk Analysis Visualizer ----------------
    with tab_chunks:
        st.subheader("🧩 Chunk Structure & Breakdown Analysis")
        st.caption("Inspect how documents were segmented into paragraph-aware chunks for embedding.")

        if not st.session_state.chunks:
            st.info("No active index available. Process documents to view chunk details.")
        else:
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Total Chunks Created", len(st.session_state.chunks))
            with col_m2:
                avg_len = sum(len(c["text"]) for c in st.session_state.chunks) // len(st.session_state.chunks)
                st.metric("Average Chunk Length", f"{avg_len} chars")
            with col_m3:
                st.metric("Target Split Size", "~1000 chars")

            st.markdown("---")
            st.subheader("Chunk Inspector")
            
            # Filter chunks by file
            file_options = list(set([c["file_name"] for c in st.session_state.chunks]))
            selected_file = st.selectbox("Filter by Source Document:", options=["All"] + file_options)

            filtered_chunks = st.session_state.chunks
            if selected_file != "All":
                filtered_chunks = [c for c in st.session_state.chunks if c["file_name"] == selected_file]

            for i, chunk in enumerate(filtered_chunks[:25], 1):
                page_str = f" | Page {chunk['page']}" if chunk['page'] is not None else ""
                with st.expander(f"Chunk #{i} - {chunk['file_name']}{page_str} ({len(chunk['text'])} chars)"):
                    st.markdown(f"<div class='chunk-card'>{chunk['text']}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
