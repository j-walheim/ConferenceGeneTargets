# %% 
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from pipeline.process_abstract import extract_abstract_info
import json
import pandas as pd
import os

from pipeline.process_abstract import process_abstracts_partition
from dotenv import load_dotenv
load_dotenv()

model = 'mistral'
# %% 
# load data/production/processed_pages/page_219.json
import json

# Load the JSON file
with open('/teamspace/studios/this_studio/ConferenceGeneTargets/data/production/processed_pages/page_6324.json') as f:
    abstract = json.load(f)

abstract_text = abstract[0].get('content')





pages_processed_dir = 'data/production/processed_pages' #os.path.join(STORAGE_DIR, ENVIRONMENT, 'processed_pages')    
pages_parsed_dir = 'data/production/parsed_pages_mistral'# os.path.join(STORAGE_DIR, ENVIRONMENT, f'parsed_pages_{model}')    
os.makedirs(pages_parsed_dir, exist_ok=True)   

import glob

for input_file in glob.glob(os.path.join(pages_processed_dir, '*.json')):
    base_name = os.path.basename(input_file)
    output_file = os.path.join(pages_parsed_dir, base_name)
    if os.path.exists(output_file):
        print(f"Skipping page {base_name} as it has already been processed.")
        continue
    
    print(f"Processing page {base_name} with model {model}...")
    with open(input_file) as f:
        abstract = json.load(f)
    abstract_text = abstract[0].get('content')
    
    # %%
    result = extract_abstract_info(abstract_text, model=model)
    abstract_dict = result.dict()
    abstract_dict['text'] = abstract_text
    abstract_dict['page_number'] = abstract[0].get('page_number')
    
    json_result = json.dumps(abstract_dict, indent=2)        
    # convert json result to string and log it
    #log_progress(json_result)
    with open(output_file, 'w') as f:
        f.write(json_result)



