import os
from typing import List
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import tempfile
import tiktoken
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
import openai
from pydantic import BaseModel

import json
from doctran import Doctran, ExtractProperty

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Set Open AI API Key
api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None, "API Key not set in the environment"

openai.api_key = api_key

# Define the PyDantic schema for contact_info
class ContactInfo(BaseModel):
    phone: str
    email: str
    experience: str
    qualifications: str

# Define the PyDantic schema for a PersonInformation
class PersonInformation(BaseModel):
    name: str
    contact_info: ContactInfo

def get_ai_response(content: str) -> PersonInformation:

    # Make a call to OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "user", "content": content}
        ],
        functions=[
            {
                "name": "get_features_from_a_cv_resume",
                "description": "Get the individual properties out of a CV/Resume",
                "parameters": {
                    'type': 'object',
                    'properties': {
                        'name': {
                            'type': 'string',
                            'description': 'Name of the person'
                        },
                        'phone': {
                            'type': 'string',
                            'description': 'Phone number of the person'
                        },
                        'email': {
                            'type': 'string',
                            'description': 'Email of the person'
                        },
                        'experience': {
                            'type': 'string',
                            'description': 'The experience of the person counted in years. Add up all the experience he has to have a single approximation. Count 2023 as the current year for calculation'
                        },
                        'qualifications': {
                            'type': 'string',
                            'description': 'The qualifications of the person'
                        }

                    }
                }
            }
        ],
        function_call={"name": "get_features_from_a_cv_resume"}
    )

    # Parse JSON output from the AI model
    output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
    return output
    # # Load JSON optionally into the PyDantic model (or) use it directly
    # person = PersonInformation(**output)

    # return person

async def extract_properties(content):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-3.5-turbo-16k"
    OPENAI_TOKEN_LIMIT = 15000

    doctran = Doctran(openai_api_key=OPENAI_API_KEY, openai_model=OPENAI_MODEL, openai_token_limit=OPENAI_TOKEN_LIMIT)
    document = doctran.parse(content=content)
    properties = [
            ExtractProperty(
                name="contact_info", 
                description="A list of each person mentioned and their contact information",
                type="array",
                items={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the person"
                        },
                        "contact_info": {
                            "type": "object",
                            "properties": {
                                "phone": {
                                    "type": "string",
                                    "description": "The phone number of the person"
                                },
                                "email": {
                                    "type": "string",
                                    "description": "The email address of the person"
                                },
                                "experience": {
                                    "type": "string",
                                    "description": "The work experience of the person, can be voluntary or professional"
                                },
                                "qualifications": {
                                    "type": "string",
                                    "description": "The qualifications of the person"
                                }
                            }
                        }
                    }
                },
                required=True
            )
    ]
    
    transformed_document = await document.extract(properties=properties).execute()
    print(json.dumps(transformed_document.extracted_properties, indent=2))
    return transformed_document.extracted_properties["contact_info"][0]

def add_vectors_to_existing_FAISS(chunked_docs,old_Knowledgebase):
    """Embeds a list of Documents and adds them to a FAISS Knowledgebase"""
    # Embed the chunks
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore
    Knowledgebase = FAISS.from_documents(chunked_docs,embeddings)
    Knowledgebase.merge_from(old_Knowledgebase)
    return Knowledgebase

def add_vectors_to_FAISS(docs):
    """Embeds a list of Documents and adds them to a FAISS Knowledgebase"""
    # Embed the chunks
    embeddings = OpenAIEmbeddings()  # type: ignore
    Knowledgebase = FAISS.from_documents(docs,embeddings)
    return Knowledgebase

def refined_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500, # You can play around with this parameter to adjust the length of each chunk
        chunk_overlap  = 10,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function = len,
    )

    # for doc in docs:
    #     doc.metadata["filename_key"] = convert_filename_to_key(doc.metadata["source"])

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

    all_content = ""
    for doc in pdf_data:
        all_content += doc.page_content

    return [Document(page_content=all_content,metadata={"source":filename})]