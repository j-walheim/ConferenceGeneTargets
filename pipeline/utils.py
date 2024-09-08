import os
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import json
from openai import AzureOpenAI
import requests
from RAG_term_normalisation.vectorstore_ontology import VectorStore
from RAG_term_normalisation.get_disease_terms import prepare_disease_synonyms
from RAG_term_normalisation.get_gene_synonyms import prepare_gene_synonyms
import re

load_dotenv()

def initialize_vectorstore():
    # Prepare disease and gene synonyms
    disease_synonyms = prepare_disease_synonyms()
    gene_synonyms = prepare_gene_synonyms()
    vectorstore = VectorStore()
    vectorstore.prepare_lookup(gene_synonyms, 'genes', 'Symbol', 'Synonyms', description_col='description')
    vectorstore.prepare_lookup(disease_synonyms, 'diseases', 'disease', 'synonym')
    vectorstore.prepare_vectorstores()
    return vectorstore

def get_llm(model='groq'):
    if model == 'mistral':
        return MistralAPI()
    elif model == "gpt-4o":
        return AzureOpenAIAPI()
    else:
        return GroqAPI()

class MistralAPI:
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.base_url = "https://api.mistral.ai/v1/chat/completions"

    def chat(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistral-large-latest",
            "messages": messages,
            "temperature": 0
        }
        response = requests.post(self.base_url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']

class AzureOpenAIAPI:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-05-01-preview"
        )
        self.deployment_name = "gpt-4o"
    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages
        )
        response_message = response.choices[0].message
        return response_message.content

class GroqAPI:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def chat(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.1-70b-versatile",
            "messages": messages,
            "temperature": 0
        }
        response = requests.post(self.base_url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']

def create_extraction_chain(prompt, llm, schema):
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = llm.chat(messages)

    # Extract content between <answer> tags
    extracted = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
    
    # Remove any empty strings from the extracted list and split by comma
    extracted = [item.strip() for item in ','.join(extracted).split(',') if item.strip()]
    
    # Return the list if it's not empty, otherwise return an empty list
    return extracted if extracted else []
