# %% 
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
#from pipeline.utils import initialize_vectorstore
from pipeline.process_target_information import extract_target_from_abstract
from RAG_term_normalisation.vectorstore_gene_synonyms import VectorStore_genes
# from pipeline.process_disease_information import extract_disease_info
import json
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()

model = 'gpt-4o'
# %% 
# Load the CSV file
import pandas as pd

abstracts_df = pd.read_csv('data/abstracts_posters_esmo.csv')
# abstracts_df = pd.read_csv('data/abstracts_20_random.csv')

pages_processed_dir = 'data/production/processed_pages'
pages_parsed_dir = f'data/production/parsed_pages_{model}'
os.makedirs(pages_parsed_dir, exist_ok=True)   

vectorstore = VectorStore_genes()

from tqdm import tqdm

for _, row in tqdm(abstracts_df.iterrows(), total=len(abstracts_df), desc="Processing abstracts"):
    page_number = row['Abstract Number']
    abstract_text = f"Title: {row['Title']}\n\n{row['Abstract']}"
    
    output_file = os.path.join(pages_parsed_dir, f'page_{page_number}.csv')
    if os.path.exists(output_file):
        continue
    
    reasoning, potential_genes, gene_context, symbols_only, reasoning_second_prompt, targets = extract_target_from_abstract(abstract_text, model=model, vectorstore=vectorstore)
    
    df = pd.DataFrame({
        'page_number': [page_number],
        'text': [abstract_text],
        'reasoning': [reasoning],
        'potential_genes': [potential_genes],
        'gene_context': [gene_context],
        'symbols_only': [symbols_only],
        'reasoning_second_prompt': [reasoning_second_prompt],
        'target': [targets],
    })
    
    df.to_csv(output_file, index=False)




# %%
