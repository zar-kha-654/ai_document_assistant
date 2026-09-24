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

# Set page layout
st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")


# ------------------------------------------------------------------------------
# 1. Caching Heavy Resources
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Embedding Model...")
def load_embedding_model():
    """Loads and caches the SentenceTransformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_groq_client():
    """Retrieves Groq client using secrets."""
    api_key = st.secrets.get("groq_api_key") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("🔑 Groq API key not found! Please configure `groq_api_key` in `.streamlit/secrets.toml`.")
        st.stop()
    return Groq(api_key=api_key)


# ------------------------------------------------------------------------------
# 2. Text Extraction Functions
# ------------------------------------------------------------------------------
def extract_text_from_pdf(file_path, file_name):
    """Extracts text page-by-page from a PDF document."""
    documents = []
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append({
                    "file_name": file_name,
                    "page": i + 1,
                    "text": text
                })
    except Exception as e:
        st.warning(f"Failed to read PDF '{file_name}': {e}")
    return documents


def extract_text_from_docx(file_path, file_name):
    """Extracts text from a DOCX document."""
    try:
        doc = DocxDocument(file_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        if text.strip():
            return [{
                "file_name": file_name,
                "page": None,
                "text": text
            }]
    except Exception as e:
        st.warning(f"Failed to read DOCX '{file_name}': {e}")
    return []


def extract_text_from_txt(file_path, file_name):
    """Extracts text from a plain TXT or Markdown file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if text.strip():
            return [{
                "file_name": file_name,
                "page": None,
                "text": text
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
# 3. Google Drive Handling (Fixed Download & Extension Resolution)
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
                # Let gdown auto-detect output filename in tmp_dir
                fpath = gdown.download(url=drive_url, output=tmp_dir + "/", quiet=True)
                
                if fpath and os.path.exists(fpath):
                    fname = os.path.basename(fpath)
                    
                    # Fallback if gdown didn't append extension
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
# 4. Text Chunking
# ------------------------------------------------------------------------------
def chunk_text(documents, chunk_size=500, overlap=100):
    """Splits extracted text into overlapping chunks preserving file/page metadata."""
    chunks = []
    for doc in documents:
        text = doc["text"]
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk_str = text[start:end]
            
            chunks.append({
                "text": chunk_str,
                "file_name": doc["file_name"],
                "page": doc["page"]
            })
            
            start += (chunk_size - overlap)
            if start >= text_length and len(chunks) > 0:
                break

    return chunks


# ------------------------------------------------------------------------------
# 5. Hybrid Search Pipeline (Vector FAISS + Keyword)
# ------------------------------------------------------------------------------
def build_vector_store(chunks, embed_model):
    """Generates embeddings and initializes FAISS vector index."""
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return index, embeddings


def keyword_search_score(query, text):
    """Calculates simple term overlap keyword score."""
    keywords = re.findall(r"\w+", query.lower())
    if not keywords:
        return 0.0
    
    text_lower = text.lower()
    matches = sum(1 for kw in set(keywords) if kw in text_lower)
    return matches / len(set(keywords))


def hybrid_search(query, chunks, index, embed_model, top_k=3, alpha=0.7):
    """Combines semantic (FAISS) and keyword search scores."""
    # 1. Semantic search
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    
    # Retrieve top candidates via FAISS
    num_candidates = min(len(chunks), top_k * 3)
    distances, indices = index.search(query_emb, num_candidates)
    
    results = []
    for idx, sem_score in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        
        # 2. Keyword score
        kw_score = keyword_search_score(query, chunk["text"])
        
        # Combined score
        combined_score = alpha * float(sem_score) + (1 - alpha) * float(kw_score)
        
        results.append({
            "chunk": chunk,
            "score": combined_score,
            "sem_score": float(sem_score),
            "kw_score": float(kw_score)
        })
    
    # Sort by hybrid score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ------------------------------------------------------------------------------
# 6. Streamlit UI & Application Lifecycle
# ------------------------------------------------------------------------------
def main():
    st.title("📄 AI Document Assistant")
    st.markdown("Upload local documents or paste Google Drive links to analyze, search, and ask questions.")

    # Initialize Session State
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "docs_processed" not in st.session_state:
        st.session_state.docs_processed = []

    embed_model = load_embedding_model()

    # Sidebar: Document Sources
    with st.sidebar:
        st.header("📂 Document Ingestion")
        
        # Local Upload
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT, MD",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True
        )
        
        # Google Drive Link
        st.markdown("---")
        st.subheader("🌐 Google Drive Link")
        drive_url = st.text_input("Paste Drive File or Folder Link:")
        
        process_btn = st.button("Process Documents", type="primary")

    # Processing Pipeline
    if process_btn:
        all_extracted_docs = []
        processed_file_names = []

        # 1. Process Local Uploads
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

        # 2. Process Google Drive Link
        if drive_url.strip():
            with st.spinner("Fetching from Google Drive..."):
                drive_docs = fetch_from_google_drive(drive_url.strip())
                all_extracted_docs.extend(drive_docs)
                processed_file_names.append("Google Drive Source")

        # 3. Chunking & Embeddings
        if all_extracted_docs:
            with st.spinner("Chunking text and generating embeddings..."):
                chunks = chunk_text(all_extracted_docs)
                faiss_idx, _ = build_vector_store(chunks, embed_model)
                
                # Persist to session state
                st.session_state.chunks = chunks
                st.session_state.faiss_index = faiss_idx
                st.session_state.docs_processed = processed_file_names
                
            st.success(f"Processing Complete! Generated **{len(chunks)}** chunks across uploaded files.")
        else:
            st.warning("No valid text extracted. Check your uploaded files or link.")

    # Status Overview
    if st.session_state.chunks:
        st.info(f"📊 **Index Active:** {len(st.session_state.chunks)} total text chunks stored in memory.")

    # Question Answering Interface
    st.markdown("### 💬 Ask Questions")
    user_query = st.text_input("Enter your question based on the document context:")

    if user_query:
        if not st.session_state.chunks or st.session_state.faiss_index is None:
            st.warning("Please upload and process documents before asking questions.")
            return

        # Perform Hybrid Search
        relevant_results = hybrid_search(
            query=user_query,
            chunks=st.session_state.chunks,
            index=st.session_state.faiss_index,
            embed_model=embed_model,
            top_k=3
        )

        # Prepare LLM Context
        context_parts = []
        for r in relevant_results:
            c = r["chunk"]
            page_info = f" (Page {c['page']})" if c['page'] is not None else ""
            context_parts.append(f"Source: {c['file_name']}{page_info}\nContent: {c['text']}")
        
        formatted_context = "\n\n---\n\n".join(context_parts)

        # Prompt Groq LLM with supported models
        system_prompt = (
            "You are a strict QA assistant. Answer the user's question using ONLY the provided document context below.\n"
            "If the information required to answer the question is not present in the context, respond with:\n"
            "'I'm sorry, but the provided document context does not contain this information.'\n"
            "Do not use outside knowledge or extrapolate beyond the text."
        )
        
        user_prompt = f"Context:\n{formatted_context}\n\nQuestion: {user_query}"

        with st.spinner("Generating answer via Groq..."):
            client = get_groq_client()
            
            # Known Groq model strings in order of preference
            available_models = [
                "llama-3.3-70b-versatile",
                "llama3-8b-8192",
                "llama3-70b-8192",
                "mixtral-8x7b-32768"
            ]
            
            answer = None
            last_err = ""

            for model_name in available_models:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0
                    )
                    answer = response.choices[0].message.content
                    break  # Success!
                except Exception as err:
                    last_err = str(err)
                    continue

            if answer:
                st.subheader("💡 Answer")
                st.write(answer)
            else:
                st.error("🔑 Groq API Key Authentication or Model Permission Error!")
                st.warning(
                    "Groq could not authenticate your key or locate active models.\n\n"
                    "**Fix Instructions:**\n"
                    "1. Get a free key at [console.groq.com](https://console.groq.com/keys)\n"
                    "2. Add it to Streamlit Secrets: **Manage app** → **Settings** → **Secrets**\n"
                    "```toml\ngroq_api_key = \"gsk_your_key_here\"\n```"
                )
                st.expander("Show detailed error").write(last_err)
                st.stop()

        # Display Answer
        st.subheader("💡 Answer")
        st.write(answer)

        # Display Retrieved Sources below the answer
        st.markdown("---")
        st.subheader("🔍 Retrieved Context & Sources")
        for idx, res in enumerate(relevant_results, 1):
            chunk = res["chunk"]
            page_str = f" | Page {chunk['page']}" if chunk['page'] is not None else ""
            
            with st.expander(f"Source {idx}: {chunk['file_name']}{page_str} (Score: {res['score']:.3f})"):
                st.write(chunk["text"])


if __name__ == "__main__":
    main()
