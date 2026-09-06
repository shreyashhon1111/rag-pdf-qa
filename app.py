import time
from datetime import datetime

import streamlit as st

import rag_core


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG PDF Q&A",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

    <style>
        :root {
            --bg: #FAFAFC;
            --panel: #FFFFFF;
            --border: #E9E9F1;
            --text: #1F2333;
            --muted: #6B7280;
            --accent-a: #6C5CE7;
            --accent-b: #17B8A6;
            --accent-soft: rgba(108, 92, 231, 0.08);
            --good: #17A673;
            --warn: #C98A1B;
            --bad: #D65C5C;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, sans-serif;
            color: var(--text);
        }

        .stApp { background: var(--bg); }
        .main .block-container { padding-top: 2rem; }

        /* HEADER */
        .app-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
            background: linear-gradient(100deg, var(--accent-a) 0%, var(--accent-b) 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: rr-gradient-shift 6s ease-in-out infinite;
            display: inline-block;
        }
        @keyframes rr-gradient-shift {
            0%, 100% { background-position: 0% center; }
            50% { background-position: 100% center; }
        }
        .app-subtitle { color: var(--muted); font-size: 1rem; margin-bottom: 1.6rem; }

        /* SIDEBAR */
        section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
        .sidebar-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.9rem; }
        .sidebar-section-label {
            font-size: 0.72rem; font-weight: 600; color: var(--muted);
            margin: 1rem 0 0.4rem 0;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            border-radius: 10px; border: 1.5px dashed var(--border);
            transition: border-color 0.2s ease, background 0.2s ease;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent-a); background: var(--accent-soft);
        }

        .doc-card {
            padding: 0.6rem 0.75rem;
            border-radius: 10px;
            background: var(--accent-soft);
            border: 1px solid rgba(108, 92, 231, 0.18);
            margin-bottom: 0.5rem;
            animation: rr-fade-up 0.3s ease-out;
        }
        .doc-name { font-weight: 600; font-size: 0.88rem; word-break: break-word; }
        .doc-meta { color: var(--muted); font-size: 0.76rem; margin-top: 0.15rem; }

        .stButton > button {
            border-radius: 8px; border: 1px solid var(--border);
            background: var(--panel); color: var(--text); font-weight: 500;
            transition: all 0.18s ease;
        }
        .stButton > button:hover {
            border-color: var(--accent-a); color: var(--accent-a);
            transform: translateY(-1px); box-shadow: 0 4px 10px rgba(108, 92, 231, 0.15);
        }

        .stAlert { border-radius: 10px; animation: rr-fade-up 0.3s ease-out; }
        @keyframes rr-fade-up {
            from { opacity: 0; transform: translateY(4px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* CHAT */
        [data-testid="stChatMessage"] { border-radius: 14px; animation: rr-fade-up 0.3s ease-out; margin-bottom: 0.4rem; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            background: var(--accent-soft); border: 1px solid rgba(108, 92, 231, 0.15);
            border-radius: 14px; padding: 0.7rem 1rem;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
            background: var(--panel); border: 1px solid var(--border);
            border-radius: 14px; padding: 0.7rem 1rem;
        }
        [data-testid="stChatInput"] { border-radius: 12px; }
        [data-testid="stChatInput"]:focus-within { box-shadow: 0 0 0 2px var(--accent-soft); }

        /* CONFIDENCE BADGE */
        .rr-badge {
            display: inline-flex; align-items: center; gap: 0.35rem;
            font-size: 0.74rem; font-weight: 600; padding: 0.15rem 0.6rem;
            border-radius: 999px; margin-bottom: 0.5rem;
        }
        .rr-badge-green { background: rgba(23, 166, 115, 0.12); color: var(--good); }
        .rr-badge-amber { background: rgba(201, 138, 27, 0.12); color: var(--warn); }
        .rr-badge-red   { background: rgba(214, 92, 92, 0.12); color: var(--bad); }
        .rr-badge-gray  { background: rgba(107, 114, 128, 0.12); color: var(--muted); }

        .rr-source {
            font-size: 0.82rem; color: var(--muted);
            border-left: 2px solid var(--border);
            padding: 0.3rem 0 0.3rem 0.6rem; margin-bottom: 0.4rem;
        }
        .rr-source b { color: var(--text); }

        /* EMPTY STATE */
        .rr-empty {
            border: 1.5px dashed var(--border); border-radius: 14px;
            padding: 2.4rem 1.6rem; text-align: center; margin-top: 0.5rem; background: var(--panel);
        }
        .rr-empty h3 { margin-bottom: 0.4rem; }
        .rr-empty p { color: var(--muted); }

        @media (prefers-reduced-motion: reduce) {
            .app-title, [data-testid="stChatMessage"], .stAlert, .doc-card { animation: none !important; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = {}       # hash -> document dict
if "active_docs" not in st.session_state:
    st.session_state.active_docs = set()  # set of hashes
if "processed_names" not in st.session_state:
    st.session_state.processed_names = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    # list of {role, content, sources?, confidence?}
if "selected_model" not in st.session_state:
    st.session_state.selected_model = rag_core.DEFAULT_MODEL


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown('<div class="sidebar-title">✦ RAG PDF Q&A</div>', unsafe_allow_html=True)
    st.write("Upload one or more documents and ask questions about their content.")

    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            key = f"{uploaded_file.name}:{uploaded_file.size}"
            if key in st.session_state.processed_names:
                continue

            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    doc = rag_core.build_document(file_bytes, uploaded_file.name)
                    st.session_state.documents[doc["hash"]] = doc
                    st.session_state.active_docs.add(doc["hash"])
                    st.session_state.processed_names.add(key)

                    note = " (loaded from cache)" if doc.get("from_cache") else ""
                    st.success(f"{uploaded_file.name} processed{note}.")
                except Exception as error:
                    st.error(f"Failed to process {uploaded_file.name}: {error}")

    if not rag_core.OCR_AVAILABLE:
        st.caption("⚠️ OCR fallback for scanned PDFs is disabled (pytesseract/poppler not installed).")

    # ------------------------------------------------------
    # Document list with active/inactive toggles
    # ------------------------------------------------------
    if st.session_state.documents:
        st.markdown('<div class="sidebar-section-label">DOCUMENTS</div>', unsafe_allow_html=True)

        for h, doc in list(st.session_state.documents.items()):
            col1, col2 = st.columns([5, 1])
            with col1:
                is_active = st.checkbox(
                    doc["name"],
                    value=(h in st.session_state.active_docs),
                    key=f"active_{h}",
                    help=f"{len(doc['chunks'])} chunks indexed",
                )
                if is_active:
                    st.session_state.active_docs.add(h)
                else:
                    st.session_state.active_docs.discard(h)
            with col2:
                if st.button("✕", key=f"remove_{h}", help="Remove this document"):
                    st.session_state.documents.pop(h, None)
                    st.session_state.active_docs.discard(h)
                    st.rerun()

    st.markdown('<div class="sidebar-section-label">MODEL</div>', unsafe_allow_html=True)
    model_label = st.selectbox(
        "Model",
        options=list(rag_core.AVAILABLE_MODELS.keys()),
        label_visibility="collapsed",
    )
    st.session_state.selected_model = rag_core.AVAILABLE_MODELS[model_label]

    score_threshold = st.slider(
        "Relevance threshold", min_value=0.0, max_value=0.6, value=0.25, step=0.05,
        help="Chunks scoring below this similarity are dropped before answering.",
    )

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col_b:
        if st.session_state.chat_history:
            transcript_lines = []
            for m in st.session_state.chat_history:
                who = "You" if m["role"] == "user" else "Assistant"
                transcript_lines.append(f"**{who}:** {m['content']}\n")
            transcript = "\n".join(transcript_lines)
            st.download_button(
                "Export",
                data=transcript,
                file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    with st.expander("Usage log"):
        rows = rag_core.get_recent_logs(limit=8)
        if not rows:
            st.caption("No queries logged yet.")
        else:
            for ts, question, model, docs, latency, top_score in rows:
                when = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                st.caption(f"`{when}` · {latency:.1f}s · {model.split('/')[-1]} — {question[:40]}")

    st.caption("Powered by FAISS + BGE embeddings + Groq")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown('<div class="app-title">✦ RAG PDF Q&A</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask questions and get cited answers straight from your documents.</div>',
    unsafe_allow_html=True,
)

active_documents = [
    st.session_state.documents[h] for h in st.session_state.active_docs
    if h in st.session_state.documents
]


# ============================================================
# EMPTY STATE
# ============================================================

if not active_documents:
    st.markdown(
        """
        <div class="rr-empty">
            <h3>Upload a document to begin</h3>
            <p>Upload a PDF, DOCX, or TXT file from the sidebar, then ask questions about its content.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# CHAT INTERFACE
# ============================================================

else:

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("confidence"):
                label, color = message["confidence"]
                st.markdown(
                    f'<span class="rr-badge rr-badge-{color}">● {label} confidence</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"Sources ({len(message['sources'])})"):
                    for s in message["sources"]:
                        page_str = f", page {s['page']}" if s.get("page") else ""
                        st.markdown(
                            f'<div class="rr-source"><b>{s["doc"]}</b>{page_str} '
                            f'· score {s["score"]:.2f}<br>{s["text"][:220]}...</div>',
                            unsafe_allow_html=True,
                        )

    query = st.chat_input("Ask something about your documents...")

    if query:
        query = query.strip()

        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                try:
                    start = time.time()

                    retrieved = rag_core.retrieve(
                        query, active_documents, k=3, score_threshold=score_threshold
                    )
                    top_score = retrieved[0]["score"] if retrieved else None
                    label, color = rag_core.confidence_label(top_score)

                    st.markdown(
                        f'<span class="rr-badge rr-badge-{color}">● {label} confidence</span>',
                        unsafe_allow_html=True,
                    )

                    if not retrieved:
                        answer = "I don't know based on the provided document."
                        st.markdown(answer)
                    else:
                        answer = st.write_stream(
                            rag_core.ask_stream(
                                query,
                                retrieved,
                                history=st.session_state.chat_history[:-1],
                                model=st.session_state.selected_model,
                            )
                        )

                    latency = time.time() - start

                    if retrieved:
                        with st.expander(f"Sources ({len(retrieved)})"):
                            for s in retrieved:
                                page_str = f", page {s['page']}" if s.get("page") else ""
                                st.markdown(
                                    f'<div class="rr-source"><b>{s["doc"]}</b>{page_str} '
                                    f'· score {s["score"]:.2f}<br>{s["text"][:220]}...</div>',
                                    unsafe_allow_html=True,
                                )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved,
                        "confidence": (label, color),
                    })

                    rag_core.log_usage(
                        question=query,
                        model=st.session_state.selected_model,
                        doc_names=[d["name"] for d in active_documents],
                        latency_sec=latency,
                        top_score=top_score,
                    )

                except Exception as error:
                    st.error(f"Error: {error}")
                    if (
                        st.session_state.chat_history
                        and st.session_state.chat_history[-1]["role"] == "user"
                    ):
                        st.session_state.chat_history.pop()