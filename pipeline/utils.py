import os
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from RAG_term_normalisation.vectorstore_ontology import VectorStore
from RAG_term_normalisation.get_disease_terms import prepare_disease_synonyms
from RAG_term_normalisation.get_gene_synonyms import prepare_gene_synonyms

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
        return ChatMistralAI(
            model='mistral-large-latest',
            temperature=0
        )
    else:
        return ChatGroq(
            model='llama-3.1-70b-versatile',
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )

def create_extraction_chain(prompt, llm, schema):
    extraction_chain = prompt | llm.with_structured_output(schema=schema)
    return extraction_chain

