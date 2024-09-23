import os
import json
import duckdb
import pandas as pd
from glob import glob

# Connect to DuckDB
conn = duckdb.connect('conference_gene_targets.db')

def process_folder(folder_path):
    json_files = glob(os.path.join(folder_path, '*.json'))
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            data.append(json.load(f))
    
    df = pd.DataFrame(data)
    table_name = os.path.basename(folder_path).replace('temporary_', '')
    
    # For the indication table
    if table_name == 'indication':
        df['Extracted Indication'] = df['Extracted Indication'].combine_first(df['indication'])
        df['Indication Subtype'] = df['Indication Subtype'].combine_first(df['subtype'])
        df = df[['Abstract Number', 'Abstract Text', 'Extracted Indication', 'Indication Subtype']]

    # For the phase table
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
        "Abstract Number",
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

# Print column names for each table
tables = ['indication', 'phase', 'gene_target', 'modality']
for table in tables:
    columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
    column_names = [col[1] for col in columns]
    print(f"\nColumns in {table} table:")
    print(", ".join(column_names))

# Close the connection
conn.close()
