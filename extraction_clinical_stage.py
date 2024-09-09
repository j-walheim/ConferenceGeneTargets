# %%
import random
import pandas as pd
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from pipeline.utils import get_llm
from dotenv import load_dotenv
import re

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

# %%
@observe(as_type="generation")
def get_phase(abstract_text):
    prompt = langfuse.get_prompt("Phase")
    msg = prompt.compile(abstract=abstract_text)

    # Get the response from the LLM
    response = llm.chat(msg)
    return response

# %%
# Run phase extraction for all elements in abstracts_df
results = []
for index, row in abstracts_df.iterrows():
    abstract_text = row['Title'] + ' ' + row['Abstract']
    abstract_number = row['Abstract Number']
    response = get_phase(abstract_text)
    
    
    # extract phase from response
    phase_match = re.search(r'\[phase\](.*?)\[/phase\]', response, re.DOTALL)
    extracted_phase = phase_match.group(1).strip() if phase_match else "Error: not found"
    
    # extract phase from response
    model_match = re.search(r'\[model\](.*?)\[/model\]', response, re.DOTALL)
    extracted_model = model_match.group(1).strip() if model_match else "n/a"
    
    results.append({
        'Abstract Number': abstract_number,
        'Abstract Text': abstract_text,
        'response': response,
        'Extracted Phase': extracted_phase,
        'Preclinical model': extracted_model
    })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Export the results to a CSV file
results_df.to_csv('phase_extraction_results.csv', index=False)

print("Phase extraction completed and results exported to 'phase_extraction_results.csv'")
