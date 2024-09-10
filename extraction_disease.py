#%%

import random
import pandas as pd
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from pipeline.utils import get_llm
from dotenv import load_dotenv
import re
import os
import json
import time
import requests
from tqdm import tqdm


# Load environment variables
load_dotenv()

# Set random seed
random.seed(1)
model = 'gpt-4o'#'groq'
llm = get_llm(model)

# Load the CSV file
abstracts_df = pd.read_csv('data/abstracts_posters_esmo.csv')

# Get random 20 rows
# abstracts_df = abstracts_df.sample(n=20)

# Initialize Langfuse
langfuse = Langfuse()

# Create temporary folder if it doesn't exist
if model == 'gpt-4o':
    temp_folder = 'data/temporary_indication_gpt4o'
else:
    temp_folder = 'data/temporary_indication_groq'
os.makedirs(temp_folder, exist_ok=True)

print('output folder: ', temp_folder)

# get unique items of abstracts_df['Topic']
list(abstracts_df['Topic'].value_counts().keys())

#%%

@observe(as_type="generation")
def get_indication(abstract_text):
    prompt = langfuse.get_prompt("GetIndication")
    msg = prompt.compile(abstract=abstract_text)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = llm.chat(msg)
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 20 * (attempt + 1)
                print(f"Error occurred: {e}. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Max retries reached. Last error: {e}")
                raise()

results = []
for index, row in tqdm(abstracts_df.iterrows(), total=len(abstracts_df), desc="Extracting indications"):
    abstract_number = row['Abstract Number']
    temp_file = os.path.join(temp_folder, f"{abstract_number}.json")
    
    if os.path.exists(temp_file):
        with open(temp_file, 'r') as f:
            result = json.load(f)
    else:
        abstract_text = row['Title'] + ' ' + row['Abstract']
        response = get_indication(abstract_text)
        
        indication_match = re.search(r'\[indication\](.*?)\[/indication\]', response, re.DOTALL)
        extracted_indication = indication_match.group(1).strip() if indication_match else "Error: not found"
        
        subtype_match = re.search(r'\[subtype\](.*?)\[/subtype\]', response, re.DOTALL)
        extracted_subtype = subtype_match.group(1).strip() if subtype_match else "n/a"
        
        result = {
            'Abstract Number': abstract_number,
            'Abstract Text': abstract_text,
            'response': response,
            'Extracted Indication': extracted_indication,
            'Indication Subtype': extracted_subtype
        }
        
        with open(temp_file, 'w') as f:
            json.dump(result, f)
    
    results.append(result)

results_df = pd.DataFrame(results)
results_df.to_csv('indication_extraction_results.csv', index=False)

print("Indication extraction completed and results exported to 'indication_extraction_results.csv'")
