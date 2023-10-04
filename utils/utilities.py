import os
from typing import List
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import tempfile
import tiktoken
import re
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS

import json
from doctran import Doctran, ExtractProperty

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the environment variable
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

def add_vectors_to_existing_FAISS(chunked_docs,old_Knowledgebase):
    """Embeds a list of Documents and adds them to a FAISS Knowledgebase"""
    # Embed the chunks
    embeddings = OpenAIEmbeddings()  # type: ignore
    Knowledgebase = FAISS.from_documents(chunked_docs,embeddings)
    Knowledgebase.merge_from(old_Knowledgebase)
    return Knowledgebase

def add_vectors_to_FAISS(chunked_docs):
    """Embeds a list of Documents and adds them to a FAISS Knowledgebase"""
    # Embed the chunks
    embeddings = OpenAIEmbeddings()  # type: ignore
    Knowledgebase = FAISS.from_documents(chunked_docs,embeddings)
    return Knowledgebase

def refined_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500, # You can play around with this parameter to adjust the length of each chunk
        chunk_overlap  = 10,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function = len,
    )

    for doc in docs:
        doc.metadata["filename_key"] = convert_filename_to_key(doc.metadata["source"])

    print(f"Lenght of docs is {len(docs)}")
    return text_splitter.split_documents(docs)

def num_tokens_from_string(chunked_docs: List[Document]) -> int:

    string = ""
    print(f"Number of vectors: \n{len(chunked_docs)}")
    for doc in chunked_docs:
        string += doc.page_content

    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def create_Knowledgebase_from_docs(docs: List[Document]) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS Knowledgebase"""
    
    # Embed the chunks
    embeddings = OpenAIEmbeddings()  # type: ignore
    
    Knowledgebase = FAISS.from_documents(docs, embeddings)

    return Knowledgebase

def parse_pdf(content,filename):
    # Assuming the content is in bytes format, save it temporarily
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name  

    pdf_loader = PyMuPDFLoader(temp_file_path)
    pdf_data = pdf_loader.load()  # Load PDF file

    for doc in pdf_data:
        # Merge hyphenated words
        doc.page_content = re.sub(r"(\w+)-\n(\w+)", r"\1\2", doc.page_content)
        # Fix newlines in the middle of sentences
        doc.page_content = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", doc.page_content.strip())
        # Remove multiple newlines
        doc.page_content = re.sub(r"\n\s*\n", "\n\n", doc.page_content)

        doc.metadata["source"] = filename

    return pdf_data