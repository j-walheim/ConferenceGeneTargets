import os
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import json
from openai import AzureOpenAI,RateLimitError
import requests
from RAG_term_normalisation.vectorstore_gene_synonyms import VectorStore_genes
import time
import re
from langfuse.decorators import observe, langfuse_context

load_dotenv()

# def initialize_vectorstore():
#     # Prepare disease and gene synonyms
#     disease_synonyms = prepare_disease_synonyms()
#     gene_synonyms = prepare_gene_synonyms()
#     vectorstore = VectorStore()
#     vectorstore.prepare_lookup(gene_synonyms, 'genes', 'Symbol', 'Synonyms', description_col='description')
#     vectorstore.prepare_lookup(disease_synonyms, 'diseases', 'disease', 'synonym')
#     vectorstore.prepare_vectorstores()
#     return vectorstore

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
        max_retries = 5
        retry_delay = 20

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages
                )
                response_message = response.choices[0].message
                return response_message.content
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    print(f"Rate limit exceeded. Attempt {attempt + 1}/{max_retries}. Waiting for {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Max retries reached. Continuing with the next task.")
                    return None  # Or a default message indicating the failure
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return None  # Or a default message indicating the failure

        return None  # In case all retries fail

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
