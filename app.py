import os
import tempfile
import streamlit as st

from config import (
    PAGE_TITLE,
    PAGE_ICON,
    SUPPORTED_FILE_TYPES,
)

from utils import (
    load_pdfs,
    split_chunks,
)

from rag import RAG


st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
)

st.title("🤖 ChemE RAG Assistant")

st.caption("Upload one or more PDFs and ask questions about them.")


if "rag" not in st.session_state:
    st.session_state.rag = RAG()

if "ready" not in st.session_state:
    st.session_state.ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


with st.sidebar:

    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
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

    rebuild = st.button(
        "Rebuild Index",
        use_container_width=True,
    )

    st.divider()

    if st.session_state.ready:

        st.success("Knowledge Base Ready")

        for file in st.session_state.uploaded_files:

            st.write(f"📄 {file}")

    else:

        st.info("No documents indexed.")


if clear_chat:

    st.session_state.messages = []

    st.session_state.rag.history = []

    st.rerun()


if rebuild:

    st.session_state.rag = RAG()

    st.session_state.messages = []

    st.session_state.ready = False

    st.session_state.uploaded_files = []

    st.rerun()


if process:

    if not uploaded_files:

        st.warning("Please upload at least one PDF.")

        st.stop()

    pdf_paths = []

    with st.spinner("Reading PDFs..."):

        for file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temp:

                temp.write(file.read())

                pdf_paths.append(temp.name)

        chunks = load_pdfs(pdf_paths)

        chunks = split_chunks(chunks)

    with st.spinner("Building knowledge base..."):

        st.session_state.rag.build(chunks)

    st.session_state.ready = True

    st.session_state.uploaded_files = [
        file.name
        for file in uploaded_files
    ]

    for path in pdf_paths:

        if os.path.exists(path):

            os.remove(path)

    st.success("Knowledge base created successfully!")

    st.rerun()


st.divider()

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            if message.get("sources"):

                with st.expander("Sources"):

                    for source in message["sources"]:

                        st.write(
                            f"📄 {source['document']} — Page {source['page']}"
                        )


if not st.session_state.ready:

    st.info(
        "Upload PDFs and click **Process Documents** to begin."
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

        try:

            with st.spinner("Thinking..."):

                answer, sources = st.session_state.rag.ask(prompt)

            st.markdown(answer)

            if sources:

                with st.expander("Sources", expanded=False):

                    displayed = set()

                    for source in sources:

                        key = (
                            source["document"],
                            source["page"],
                        )

                        if key in displayed:
                            continue

                        displayed.add(key)

                        st.markdown(
                            f"📄 **{source['document']}** — Page {source['page']}"
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as e:

            st.error(str(e))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Error: {e}",
                    "sources": [],
                }
            )
