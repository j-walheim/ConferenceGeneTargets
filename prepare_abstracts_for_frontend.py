import pandas as pd
import json
import re

# Read the JSONL file
data = []
with open('final_merged_data_6.jsonl', 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)


# Function to extract abstract title
def extract_abstract_title(text):
    if not isinstance(text, str):
        return ""
    match = re.search(r'-\s*(.*?)\s*Background', text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

# Select Abstract Number, Abstract Text, Extracted Indication, Extracted Phase, potential_genes, target, and Preclinical model
selected_df = df[['Abstract Number', 'Abstract Text', 'Extracted Indication', 'Extracted Phase', 'potential_genes', 'target', 'Preclinical model']].copy()

# Extract abstract title
selected_df.loc[:, 'Abstract Title'] = selected_df['Abstract Text'].apply(extract_abstract_title)

# Function to safely evaluate the target column
def safe_eval_len(x):
    if isinstance(x, str):
        try:
            return len(eval(x))
        except:
            return 0
    elif isinstance(x, list):
        return len(x)
    else:
        return 0

# Only keep entries that have are not an empty list in the target column
selected_df = selected_df[selected_df['target'].apply(safe_eval_len) > 0]

# Function to handle HER2 and ERBB2 synonyms
def handle_her2_erbb2(genes):
    gene_list = eval(genes)
    if "HER2" in gene_list and "ERBB2" not in gene_list:
        gene_list = [gene.replace("HER2", "ERBB2") for gene in gene_list]
    elif "HER2" in gene_list and "ERBB2" in gene_list:
        gene_list = [gene for gene in gene_list if gene != "HER2"]
    return str(gene_list)

# Apply the function to the 'potential_genes' column
selected_df['potential_genes'] = selected_df['potential_genes'].apply(handle_her2_erbb2)

# Function to combine Phase and Preclinical model
def combine_phase_and_model(row):
    if row['Extracted Phase'].lower() == 'preclinical':
        return f"Preclinical ({row['Preclinical model']})"
    return row['Extracted Phase']

# Apply the function to create a new 'Phase' column
selected_df['Phase'] = selected_df.apply(combine_phase_and_model, axis=1)

# New function to standardize phase
def standardize_phase(phase):
    if not isinstance(phase, str):
        return "Unknown"
    
    # Convert to title case
    phase = phase.title()
    
    # Replace Roman numerals with Arabic numerals
    roman_to_arabic = {'I': '1', 'Ii': '2', 'Iii': '3', 'Iv': '4', 'V': '5'}
    for roman, arabic in roman_to_arabic.items():
        phase = re.sub(rf'\b{roman}\b', arabic, phase, flags=re.IGNORECASE)
    
    # Ensure "Phase" is capitalized and has consistent spacing
    phase = re.sub(r'(?i)\bphase\s*', 'Phase ', phase)
    
    # Ensure slash-separated phases are consistent
    phase = re.sub(r'(\d[a-z]?)\s*/\s*(\d[a-z]?)', r'\1/Phase \2', phase)
    
    # Remove "And" between phases
    phase = re.sub(r'\s+And\s+', '/', phase)
    
    return phase

# Apply the standardize_phase function to the 'Phase' column
selected_df['Phase'] = selected_df['Phase'].apply(standardize_phase)

# Rename columns to make them more readable
selected_df.rename(columns={
    'Abstract Number': 'Abstract Number',
    'Extracted Indication': 'Indication',
    'potential_genes': 'Genes (incl. biomarkers)',
    'target': 'Targets',
    'Abstract Title': 'Abstract Title'
}, inplace=True)

# Remove the original 'Extracted Phase', 'Preclinical model', and 'Abstract Text' columns
selected_df = selected_df.drop(columns=['Extracted Phase', 'Preclinical model', 'Abstract Text'])

# export to pass onto frontend
selected_df.to_csv('data/esmo_data_for_frontend.csv', index=False)