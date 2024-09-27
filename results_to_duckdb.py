## Export results to duckdb and csv - for the interface we only use rows with a gene target

import os
import json
import duckdb
import pandas as pd
from glob import glob
import re

# Remove old file and connect to DuckDB
os.remove('conference_gene_targets.db')
conn = duckdb.connect('conference_gene_targets.db')

def process_folder(folder_path):
    json_files = glob(os.path.join(folder_path, '*.json'))
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            data.append(json.load(f))
    
    df = pd.DataFrame(data)
    table_name = os.path.basename(folder_path).replace('temporary_', '')
    
    if table_name == 'indication':
        df['Extracted Indication'] = df['Extracted Indication'].combine_first(df['indication'])
        df['Indication Subtype'] = df['Indication Subtype'].combine_first(df['subtype'])
        df = df[['Abstract Number', 'Abstract Text', 'Extracted Indication', 'Indication Subtype']]

    if table_name == 'phase':
        df['Extracted Phase'] = df['Extracted Phase'].combine_first(df['phase'])
        df['Preclinical model'] = df['Preclinical model'].combine_first(df['model'])
        df = df[['Abstract Number', 'Abstract Text', 'Extracted Phase', 'Preclinical model']]    
    
    columns = ', '.join([f'"{col}" VARCHAR' for col in df.columns])
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
    
    conn.register('temp_df', df)
    conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
    
    print(f"Processed {table_name}: {len(df)} rows inserted")

folders = [
    'data/temporary_indication',
    'data/temporary_phase',
    'data/temporary_gene_target',
    'data/temporary_modality',
    'data/results_posters/temporary_gene_target_old'
]

for folder in folders:
    process_folder(folder)

# Rename page_number to Abstract Number in gene_target_old
conn.execute("""
    ALTER TABLE gene_target_old
    RENAME COLUMN page_number TO "Abstract Number"
""")

# Create a new combined table
conn.execute("""
    CREATE TABLE combined_gene_target AS
    SELECT 
        COALESCE(gt."Abstract Number", gto."Abstract Number") AS "Abstract Number",
        COALESCE(gt."Abstract Text", gto.text) AS "Abstract Text",
        COALESCE(gt.potential_genes, gto.potential_genes) AS potential_genes,
        COALESCE(gt.initial_reasoning, '') AS initial_reasoning,
        COALESCE(gt.gene_context, gto.gene_context) AS gene_context,
        COALESCE(gt.symbols_only, gto.symbols_only) AS symbols_only,
        COALESCE(gt.reasoning, gto.reasoning) AS reasoning,
        COALESCE(gt.gene_target, gto.gene_target) AS gene_target
    FROM gene_target gt
    FULL OUTER JOIN gene_target_old gto ON gt."Abstract Number" = gto."Abstract Number"
""")

# Drop the original tables
conn.execute("DROP TABLE gene_target")
conn.execute("DROP TABLE gene_target_old")

# Rename the combined table to gene_target
conn.execute("ALTER TABLE combined_gene_target RENAME TO gene_target")

# Get all table names
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
table_names = [table[0] for table in tables]

# Print column names for each table
for table in table_names:
    columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
    column_names = [col[1] for col in columns]
    print(f"\nColumns in {table} table:")
    print(", ".join(column_names))


for table in table_names:
    df = conn.execute(f"SELECT * FROM {table}").fetchdf()
    df.to_csv(f'{table}_table.csv', index=False)
    print(f"Exported {table} table to {table}_table.csv")


def extract_abstract_title(text):
    if not isinstance(text, str):
        return ""
    match = re.search(r'-\s*(.*?)\s*Background', text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

import ast

def handle_her2_erbb2(genes):
    if not isinstance(genes, str) or genes.strip() == '':
        return "[]"
    try:
        gene_list = ast.literal_eval(genes)
    except:
        gene_list = [gene.strip() for gene in genes.split(';') if gene.strip()]
    
    if not isinstance(gene_list, list):
        gene_list = [str(gene_list)]
    
    gene_list = [item if isinstance(item, str) else ' '.join(map(str, item)) for item in gene_list]
    gene_list = [gene.replace('HER2', 'ERBB2') for gene in gene_list]
    gene_list = list(dict.fromkeys(gene_list))  # Remove duplicates while preserving order
    return str(gene_list)




def standardize_phase(phase):
    if not isinstance(phase, str):
        return "Unknown"
    
    phase = phase.title()
    lowerPhase = phase.lower()
    
    if 'phase 1/phase 2a' in lowerPhase or 'phase 1b/2' in lowerPhase:
        return 'Phase 1/Phase 2'
    
    if 'phase 2/3' in lowerPhase:
        return 'Phase 2/Phase 3'
    
    roman_to_arabic = {'I': '1', 'Ii': '2', 'Iii': '3', 'Iv': '4', 'V': '5'}
    for roman, arabic in roman_to_arabic.items():
        phase = re.sub(rf'\b{roman}\b', arabic, phase, flags=re.IGNORECASE)
    
    phase = re.sub(r'(?i)\bphase\s*', 'Phase ', phase)
    phase = re.sub(r'(\d[a-z]?)\s*/\s*(\d[a-z]?)', r'\1/Phase \2', phase)
    phase = re.sub(r'\s+And\s+', '/', phase)
    
    return phase

# Create a new table with the desired columns, overwriting if it exists
conn.execute("DROP TABLE IF EXISTS final_table")
conn.execute("""
CREATE TABLE final_table AS
SELECT
    i."Abstract Number",
    i."Extracted Indication" AS Indication,
    gt.potential_genes AS "Genes (incl. biomarkers)",
    gt.gene_target AS Targets,
    SUBSTRING(i."Abstract Text", 1, POSITION('Background' IN i."Abstract Text")) AS "Abstract Title",
    CASE
        WHEN LOWER(p."Extracted Phase") = 'preclinical' THEN 'Preclinical (' || p."Preclinical model" || ')'
        ELSE p."Extracted Phase"
    END AS Phase,
    m.modality AS Modality,
    m.therapeutic_agents AS "Therapeutic Agents",
    i."Abstract Text" AS "Full Abstract Text"
FROM indication i
LEFT JOIN gene_target gt ON i."Abstract Number" = gt."Abstract Number"
LEFT JOIN phase p ON i."Abstract Number" = p."Abstract Number"
LEFT JOIN modality m ON i."Abstract Number" = m."Abstract Number"
WHERE LENGTH(gt.gene_target) > 2
""")

# Fetch the data from the final_table
result = conn.execute("SELECT * FROM final_table").fetchdf()

# Apply cleaning functions
result['Abstract Title'] = result['Full Abstract Text'].apply(extract_abstract_title)
result['Genes (incl. biomarkers)'] = result['Genes (incl. biomarkers)'].apply(handle_her2_erbb2)
result['Targets'] = result['Targets'].apply(handle_her2_erbb2)
result['Phase'] = result['Phase'].apply(standardize_phase)

# Remove unnecessary columns
result = result.drop(columns=['Full Abstract Text'])

# Write the result to a CSV file
result.to_csv('final_abstracts.csv', index=False)

# Close the connection
conn.close()

print("Final table created, cleaned, and saved to 'final_abstracts.csv'")
