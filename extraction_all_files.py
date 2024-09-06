# %% 
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from pipeline.process_target_information import extract_target_from_abstract
# from pipeline.process_disease_information import extract_disease_info
import json
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()

model = 'groq'
# %% 
# Load the CSV file
import pandas as pd

abstracts_df = pd.read_csv('data/abstracts_20.csv')

pages_processed_dir = 'data/production/processed_pages'
pages_parsed_dir = f'data/production/parsed_pages_{model}'
os.makedirs(pages_parsed_dir, exist_ok=True)   

for _, row in abstracts_df.iterrows():
    page_number = row['page_number']
    abstract_text = row['content']
    
    output_file = os.path.join(pages_parsed_dir, f'page_{page_number}.json')
    if os.path.exists(output_file):
        print(f"Skipping page {page_number} as it has already been processed.")
        continue
    
    print(f"Processing page {page_number} with model {model}...")
    
    # %%
    result = extract_target_from_abstract(abstract_text, model=model)
    abstract_dict = result.dict()
    abstract_dict['text'] = abstract_text
    abstract_dict['page_number'] = page_number
    
    json_result = json.dumps(abstract_dict, indent=2)        
    with open(output_file, 'w') as f:
        f.write(json_result)




# %%
