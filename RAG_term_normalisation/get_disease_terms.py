# %%
import requests
import re
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
        entries.append(term)  # Add the primary term to entries
        return entries
    else:
        return [term]  # Return the primary term if no synonyms found
 # %%
# Usage example

def prepare_disease_synonyms():
    df_disease = pd.read_csv('data/RAG_LLM/features_raw/cancer_types.csv')
    
    disease_synonyms = []
    
    for _, row in df_disease.iterrows():
        disease_term = row['Cancer_Type'] 
        synonyms = get_mesh_synonyms(disease_term)
        for synonym in synonyms:
            disease_synonyms.append({'disease': disease_term, 'synonym': synonym})
    df_synonyms = pd.DataFrame(disease_synonyms)
    
    df_synonyms.to_csv('data/RAG_LLM/features/disease_synonyms.csv', index=False)

    return df_synonyms