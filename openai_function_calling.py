import openai
import os
import json
from dotenv import load_dotenv

from pydantic import BaseModel

load_dotenv()

# Set Open AI API Key
api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None, "API Key not set in environment"

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

# Make a call to OpenAI
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
       {"role": "user", "content": f""}
    ],
    functions=[
        {
            "name": "get_features_from_a_cv_resume",
            "description": "Get the individual properties out of a CV/Resume",
            "parameters": PersonInformation.schema()  # Use the PersonInformation schema here
        }
    ],
    function_call={"name": "get_features_from_a_cv"}
)

# Parse JSON output from the AI model
output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])

print(output)

# Load JSON optionally into the PyDantic model (or) use it directly
person = PersonInformation(**output)

print(person)
