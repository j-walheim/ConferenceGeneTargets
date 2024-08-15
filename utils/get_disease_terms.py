# %%
import requests
import re
from fuzzywuzzy import process
import pandas as pd
 
def get_mesh_synonyms(term):
    url = f'https://www.ncbi.nlm.nih.gov/mesh/?term=%22{requests.utils.quote(term)}%22%5BMeSH%20Terms%5D&cmd=DetailsSearch'
    response = requests.get(url)
    content = response.text
    pattern = r'Entry Terms.*?<ul>(.*?)</ul>'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()
        entries = re.findall(r'<li>(.*?)</li>', extracted_text)
        # Filter out entries with HTML tags
        entries = [entry for entry in entries if '<' not in entry and '>' not in entry]
        return entries
    else:
        return []

 # %%
# Usage example

def get_disease_synonyms():
    df_disease = pd.read_csv('data/RAG_LLM/features_raw/cancer_types.csv')
    
    disease_synonyms = {}
    
    for _, row in df_disease.iterrows():
        disease_term = row['cancer_type']  # Assuming 'cancer_type' is the column name
        synonyms = get_mesh_synonyms(disease_term)
        disease_synonyms[disease_term] = synonyms
    
    import json
    with open('data/RAG_LLM/features/disease_synonyms.json', 'w') as f:
        json.dump(disease_synonyms, f, indent=2)
