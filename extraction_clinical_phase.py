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

# Load environment variables
load_dotenv()

# Set random seed
random.seed(1)
model = 'gpt-4o'
llm = get_llm(model)

# Load the CSV file
abstracts_df = pd.read_csv('data/abstracts_posters_esmo.csv')

# Get random 20 rows
#abstracts_df = abstracts_df.sample(n=20)

# Initialize Langfuse
langfuse = Langfuse()

# Create temporary folder if it doesn't exist
temp_folder = 'data/temporary_stage'
os.makedirs(temp_folder, exist_ok=True)



@observe(as_type="generation")
def get_phase(abstract_text):
    prompt = langfuse.get_prompt("Phase")
    msg = prompt.compile(abstract=abstract_text)

    while True:
        try:
            # Get the response from the LLM
            response = llm.chat(msg)
            return response
        except requests.exceptions.RequestException as e:
            if "openai.RateLimitError" in str(e):
                print("Rate limit exceeded. Waiting for 5 seconds before retrying...")
                time.sleep(20)
            else:
                raise
from tqdm import tqdm

# Run phase extraction for all elements in abstracts_df
results = []
for index, row in tqdm(abstracts_df.iterrows(), total=len(abstracts_df), desc="Extracting clinical stages"):
    abstract_number = row['Abstract Number']
    temp_file = os.path.join(temp_folder, f"{abstract_number}.json")
    
    if os.path.exists(temp_file):
        with open(temp_file, 'r') as f:
            result = json.load(f)
    else:
        abstract_text = row['Title'] + ' ' + row['Abstract']
        response = get_phase(abstract_text)
        
        phase_match = re.search(r'\[phase\](.*?)\[/phase\]', response, re.DOTALL)
        extracted_phase = phase_match.group(1).strip() if phase_match else "Error: not found"
        
        model_match = re.search(r'\[model\](.*?)\[/model\]', response, re.DOTALL)
        extracted_model = model_match.group(1).strip() if model_match else "n/a"
        
        result = {
            'Abstract Number': abstract_number,
            'Abstract Text': abstract_text,
            'response': response,
            'Extracted Phase': extracted_phase,
            'Preclinical model': extracted_model
        }
        
        with open(temp_file, 'w') as f:
            json.dump(result, f)
    
    results.append(result)
# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Export the results to a CSV file
results_df.to_csv('phase_extraction_results.csv', index=False)

print("Phase extraction completed and results exported to 'phase_extraction_results.csv'")
