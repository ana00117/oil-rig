"""Streamlit interface for the Oil Rig Data RAG chatbot."""

import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Oil Rig Data Assistant", page_icon="🛢️")
st.title("🛢️ Oil Rig Data Assistant")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files (drilling reports, manuals, logs, safety docs)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write("Uploaded files:")
        for f in uploaded_files:
            st.write(f"- {f.name}")

    if st.button("Process Documents", disabled=not uploaded_files):
        with st.spinner("Extracting, chunking, and indexing..."):
            try:
                pipeline = RAGPipeline()
                num_docs, num_chunks = pipeline.process_documents(uploaded_files)
                st.session_state.pipeline = pipeline
                st.session_state.docs_processed = True
                st.success(f"Indexed {num_docs} pages into {num_chunks} chunks.")
            except Exception as e:
                st.error(f"Error processing documents: {e}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.caption(
    "⚠️ Answers are generated from your uploaded rig documents. "
    "Verify safety-critical or operational information against the source before acting on it."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s['source']}, page {s['page']}")

question = st.chat_input("Ask a question about your rig documents...")

if question:
    if not st.session_state.docs_processed:
        st.warning("Please upload and process at least one PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = st.session_state.pipeline.ask(question)
                except Exception as e:
                    answer, sources = f"Error: {e}", []
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.write(f"- {s['source']}, page {s['page']}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
