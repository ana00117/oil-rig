
import streamlit as st
from pathlib import Path
import tempfile

from config import (
    APP_TITLE,
    PAGE_ICON,
    MEMORY_SIZE,
)

from document_loader import DocumentLoader
from vector_store import VectorStoreManager
from retriever import HybridRetriever
from memory import ConversationMemory
from rag_pipeline import RAGPipeline


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
)

st.title(APP_TITLE)


if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "documents" not in st.session_state:
    st.session_state.documents = []


with st.sidebar:

    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process = st.button(
        "Process Documents",
        use_container_width=True,
    )

    rebuild = st.button(
        "Rebuild Index",
        use_container_width=True,
    )

    clear_chat = st.button(
        "Clear Chat",
        use_container_width=True,
    )

    st.divider()

    if st.session_state.indexed:

        st.success("Knowledge base ready")

        st.metric(
            "Chunks",
            len(st.session_state.documents),
        )

    else:

        st.info("No documents indexed")


if clear_chat:

    st.session_state.messages = []

    if st.session_state.pipeline is not None:
        st.session_state.pipeline.clear_memory()

    st.rerun()


if rebuild:

    st.session_state.pipeline = None
    st.session_state.documents = []
    st.session_state.indexed = False

    st.rerun()


if process:

    if not uploaded_files:

        st.warning("Upload at least one PDF.")

        st.stop()

    with st.spinner("Building knowledge base..."):

        temp_paths = []

        for uploaded in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as tmp:

                tmp.write(uploaded.read())

                temp_paths.append(tmp.name)

        loader = DocumentLoader()

        documents = loader.load_and_split(
            temp_paths
        )

        vector_store = VectorStoreManager()

        vector_store.create_vector_store(
            documents
        )

        vector_store.save()

        retriever = HybridRetriever(
            vector_store_manager=vector_store,
            documents=documents,
        )

        memory = ConversationMemory(
            max_messages=MEMORY_SIZE
        )

        pipeline = RAGPipeline(
            retriever=retriever,
            memory=memory,
        )

        st.session_state.pipeline = pipeline

        st.session_state.documents = documents

        st.session_state.indexed = True

    st.success("Documents processed successfully.")

    st.rerun()


st.divider()

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("Sources"):

                for source in message["sources"]:

                    st.write(
                        f"**{source['document']}** — Page {source['page']}"
                    )


if not st.session_state.indexed:

    st.info("Upload PDFs and click **Process Documents** to begin.")

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

        with st.spinner("Thinking..."):

            try:

                result = st.session_state.pipeline.ask(prompt)

                answer = result["answer"]

                sources = result["sources"]

                placeholder.markdown(answer)

                if sources:

                    with st.expander("Sources", expanded=False):

                        for source in sources:

                            st.markdown(
                                f"**{source['document']}** — Page {source['page']}"
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as e:

                error = f"Error: {e}"

                placeholder.error(error)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error,
                        "sources": [],
                    }
                )
