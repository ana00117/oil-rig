import os
import tempfile

import streamlit as st

from config import (
    PAGE_TITLE,
    PAGE_ICON,
    SUPPORTED_FILE_TYPES,
)

from document_loader import DocumentLoader
from rag_pipeline import RAGPipeline


st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
)

st.title("🤖 ChemE RAG Assistant")
st.caption("Upload PDF documents and ask questions from them.")


if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = (
        st.session_state.pipeline.vector_store.exists()
    )

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


with st.sidebar:

    st.header("📂 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
    )

    process = st.button(
        "Process Documents",
        use_container_width=True,
    )

    clear_chat = st.button(
        "Clear Chat",
        use_container_width=True,
    )

    st.divider()

    if st.session_state.documents_loaded:

        st.success("Knowledge Base Ready")

        if st.session_state.uploaded_files:

            st.write("Indexed Files")

            for file in st.session_state.uploaded_files:

                st.write(f"📄 {file}")

    else:

        st.info("Upload documents to begin.")


if clear_chat:

    st.session_state.messages = []

    st.session_state.pipeline.chat_history = []

    st.rerun()


if process:

    if not uploaded_files:

        st.warning("Please upload at least one PDF.")

        st.stop()

    pdf_paths = []

    with st.spinner("Processing documents..."):

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temp:

                temp.write(uploaded_file.read())

                pdf_paths.append(temp.name)

        loader = DocumentLoader()

        documents = loader.load_pdfs(pdf_paths)

        st.session_state.pipeline.build_vector_store(
            documents
        )

        st.session_state.documents_loaded = True

        st.session_state.uploaded_files = [
            file.name
            for file in uploaded_files
        ]

    for path in pdf_paths:

        if os.path.exists(path):

            os.remove(path)

    st.success("Documents processed successfully.")

    st.rerun()


st.divider()

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            with st.expander("Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"**📄 {source['document']}** (Page {source['page']})"
                    )

                    st.caption(source["text"])


if not st.session_state.documents_loaded:

    st.info(
        "Upload PDF documents and click **Process Documents**."
    )

    st.stop()
prompt = st.chat_input("Ask a question about your documents...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        try:

            with st.spinner("Thinking..."):

                answer, sources = (
                    st.session_state.pipeline.ask(prompt)
                )

            placeholder.markdown(answer)

            if sources:

                with st.expander(
                    "📚 Sources",
                    expanded=False,
                ):

                    for source in sources:

                        st.markdown(
                            f"**📄 {source['document']}**"
                        )

                        st.write(
                            f"**Page:** {source['page']}"
                        )

                        st.caption(
                            source["text"]
                        )

                        st.divider()

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as e:

            placeholder.error(
                f"Error: {str(e)}"
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Error: {str(e)}",
                    "sources": [],
                }
            )
