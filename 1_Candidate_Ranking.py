import streamlit as st
from utils.utilities import (
    refined_docs,
    parse_pdf,
    num_tokens_from_string,
    add_vectors_to_FAISS
)

from dotenv import load_dotenv

st.set_page_config(
    page_title='Prodigy AI',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Load environment variables from .env file
load_dotenv()

st.title("Create New KnowledgeBase")

accepted_file_types = ["pdf"]

uploaded_files = st.file_uploader("Upload one or more files", accept_multiple_files=True, type=accepted_file_types)

try:
    if st.button("Create Knowledgebase", use_container_width=True):
        if uploaded_files:
            docs = None
            tot_len = 0

            for file in uploaded_files:
                file_extension = file.name.split(".")[-1].upper()
                st.write(f'File: {file.name}, Extension: {file_extension}')
                file_content = file.read()  # Read the content of the uploaded file

                if file_extension == 'PDF':
                    if docs is None:
                        docs = parse_pdf(file_content,filename=file.name)
                    else:
                        docs = docs + parse_pdf(file_content,filename=file.name)

                else:
                    raise ValueError("File type not supported!")

            chunked_docs = refined_docs(docs)

            no_of_tokens = num_tokens_from_string(chunked_docs)
            st.write(f"Number of tokens: \n{no_of_tokens}")

            if no_of_tokens:
                with st.spinner("Creating Knowledgebase..."):
                    st.session_state.Knowledgebase = add_vectors_to_FAISS(chunked_docs=chunked_docs)
            else:
                st.error("No text found in the docs to index. Please make sure the documents you uploaded have a selectable text.")
        else:
            st.error("Please add some files first!")
except Exception as e:
    st.error(f"An error occured while indexing your documents: {e}\n\nPlease fix the error and try again.")