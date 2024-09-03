import json
import sqlite3
import os
import glob
import pandas as pd
# Set the directory path for the parsed pages
parsed_pages_dir = 'data/production/parsed_pages_mistral'

# Set the path for the SQLite database file
db_path = 'data/production/parsed_pages.db'

# Create a connection to the SQLite database (it will create a new file if it doesn't exist)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the table to store the parsed page data
cursor.execute('''
    CREATE TABLE IF NOT EXISTS parsed_pages (
        abstract_number TEXT,
        title TEXT,
        authors TEXT,
        disease TEXT,
        gene TEXT,
        gene_target TEXT,
        organism TEXT,
        trial_stage TEXT,
        compound_name TEXT,
        abstract_category TEXT,
        text TEXT,
        page_number INTEGER
    )
''')

# Iterate over the parsed page files and insert the data into the database
for file_path in glob.glob(os.path.join(parsed_pages_dir, 'page_*.json')):
    with open(file_path) as f:
        data = json.load(f)
        
        # Extract the values from the JSON data
        abstract_number = data.get('abstract_number', '')
        title = data.get('title', '')
        
        authors_value = data.get('authors', [])
        authors = ';'.join([author['name'] for author in authors_value]) if isinstance(authors_value, list) else authors_value.get('name', '') if authors_value else ''
        
        disease_value = data.get('disease', [])
        disease = ';'.join(disease_value) if isinstance(disease_value, list) else disease_value if disease_value else ''
        
        gene_value = data.get('gene', [])
        gene = ';'.join(gene_value) if isinstance(gene_value, list) else gene_value if gene_value else ''
        
        gene_target = data.get('gene_target', '')
        
        organism_value = data.get('organism', [])
        organism = ';'.join(organism_value) if isinstance(organism_value, list) else organism_value if organism_value else ''
        
        trial_stage = data.get('trial_stage', '')
        compound_name = data.get('compound_name', '')
        abstract_category = data.get('abstract_category', '')
        text = data.get('text', '')
        page_number = data.get('page_number', '')
        
        # Insert the data into the database
        cursor.execute('''
            INSERT INTO parsed_pages (
                abstract_number, title, authors, disease, gene, gene_target, organism,
                trial_stage, compound_name, abstract_category, text, page_number
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (abstract_number, title, authors, disease, gene, gene_target, organism,
              trial_stage, compound_name, abstract_category, text, page_number))

# Commit the changes and close the connection
conn.commit()

# Load the data from the database into a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM parsed_pages", conn)

# Show the head of the DataFrame
print("Head of the parsed_pages table:")
print(df.head())

conn.close()
