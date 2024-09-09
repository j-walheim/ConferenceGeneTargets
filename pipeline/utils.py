import os
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import json
from openai import AzureOpenAI
import requests
from RAG_term_normalisation.vectorstore_gene_synonyms import VectorStore_genes

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
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages
                )
                response_message = response.choices[0].message
                return response_message.content
            except requests.exceptions.RequestException as e:
               # if "openai.RateLimitError" in str(e):
                print("Rate limit exceeded. Waiting for 20 seconds before retrying...")
                time.sleep(20)
                #else:
                #    raise


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

def create_extraction_chain(prompt, llm):
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = llm.chat(messages)

    reasoning = re.findall(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    # Extract content between <answer> tags
    extracted = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
    
    # Remove any empty strings from the extracted list and split by comma
    extracted = [item.strip() for item in ','.join(extracted).split(',') if item.strip()]
    
    # Prepare the result dictionary
    result = {
        "reasoning": reasoning if reasoning else "",
        "extracted": extracted if extracted else []
    }
    
    # Return the dictionary containing both reasoning and extracted items
    return result
