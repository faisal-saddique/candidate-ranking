import streamlit as st
from utils.utilities import (
    refined_docs,
    parse_pdf,
    num_tokens_from_string,
    add_vectors_to_FAISS,
    extract_properties
)
from dotenv import load_dotenv
import asyncio
import pandas as pd

# Load environment variables from .env file
load_dotenv()
import os



st.set_page_config(
    page_title='Prodigy AI',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded'
)



async def main():

    col1, col2 = st.columns(2,gap="large")

    with col1:
        st.title("CANDIDATE *RANKING*")

        accepted_file_types = ["pdf"]

        uploaded_files = st.file_uploader("Upload one or more files", accept_multiple_files=True, type=accepted_file_types)

        job_description = st.text_area(label="Please enter the job description here:", height=150)

        if st.button("Proceed", use_container_width=True) and job_description:
            if uploaded_files:
                docs = None
                extracted_features_list = []
                for file in uploaded_files:
                    file_extension = file.name.split(".")[-1].upper()
                    st.write(f'File: {file.name}, Extension: {file_extension}')
                    file_content = file.read()  # Read the content of the uploaded file

                    if file_extension == 'PDF':
                        retrieved_doc = parse_pdf(file_content, filename=file.name)
                        if docs is None:
                            docs = retrieved_doc
                        else:
                            docs = docs + retrieved_doc
                    else:
                        raise ValueError("File type not supported!")
                    
                no_of_docs = len(docs)
                no_of_tokens = num_tokens_from_string(docs)
                st.write(f"Number of tokens: \n{no_of_tokens}")

                if no_of_tokens:
                    with st.spinner("Creating faiss_docstore..."):
                        docstore = add_vectors_to_FAISS(docs=docs)
                        docs_and_scores = docstore.similarity_search_with_score(job_description,k=no_of_docs)
                        for item in docs_and_scores:
                            # Define your list of dictionaries
                            extracted_features_list.append({"properties":await extract_properties(content=item[0].page_content),"score":item[1]})

                        # Extract data and create a Pandas DataFrame
                        df = pd.DataFrame([
                            {
                                'Name': entry['properties']['name'],
                                # 'Phone': entry['properties']['contact_info']['phone'],
                                # 'Email': entry['properties']['contact_info']['email'],
                                'Ranking':entry['score'],
                                'Experience': entry['properties']['contact_info']['experience'],
                                'Qualifications': entry['properties']['contact_info']['qualifications']
                            }
                            for entry in extracted_features_list
                        ])
                        # Use Streamlit to display the DataFrame
                        st.write(df)
                else:
                    st.error("No text found in the docs to index. Please make sure the documents you uploaded have selectable text.")
            else:
                st.error("Please add some files first!")

    with col2:
        st.image("./assets/Picture1.png")

if __name__ == "__main__":
    asyncio.run(main())