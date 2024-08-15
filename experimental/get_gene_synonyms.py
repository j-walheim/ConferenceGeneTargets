# %%
import requests
 
from biomart import BiomartServer
import pandas as pd
from dagster import asset
import os

import pandas as pd
import os
from biomart import BiomartServer
import pandas as pd

def get_hugo_symbols_df():
    # Connect to the Ensembl Biomart server
    server = BiomartServer("http://www.ensembl.org/biomart")
    # Select the dataset
    dataset = server.datasets['hsapiens_gene_ensembl']
#    available_attributes = dataset.attributes           
#    attributes = available_attributes.keys()#[attr.name for attr in available_attributes]

    attributes = ['ensembl_gene_id', 'external_gene_name']

    response = dataset.search({'attributes': attributes})

    # Process the results
    ensg_list = []
    hugo_list = []
    for line in response.iter_lines():
        line = line.decode('utf-8').strip()
        if not line:  # Skip empty lines
            continue
        parts = line.split('\t')
        if len(parts) == 2:
            ensg, hugo = parts
            if hugo:  # Only add non-empty Hugo symbols
                ensg_list.append(ensg)
                hugo_list.append(hugo)
    
    # Create a DataFrame using pandas
    df = pd.DataFrame({
        'ENSG': ensg_list,
        'HUGO': hugo_list
    })
    
    return df 
 
 
 
 
df_genes = get_hugo_symbols_df()
 
 # %%
 
# download full ncbi gene info https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz
import gzip
import shutil
import requests
import os


# URL of the gene_info.gz file
url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz"

# Local filenames
gz_file = "gene_info.gz"
extracted_file = "gene_info"

# Download the file
print("Downloading gene_info.gz...")
response = requests.get(url, stream=True)
with open(gz_file, 'wb') as f:
    shutil.copyfileobj(response.raw, f)

# Extract the gzipped file
print("Extracting gene_info...")
with gzip.open(gz_file, 'rb') as f_in:
    with open(extracted_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Remove the gzipped file
os.remove(gz_file)

print(f"Extraction complete. File saved as {extracted_file}")


# %%


import pandas as pd

# File path
file_path = 'gene_info'

# Columns we want to retrieve
columns_to_use = ['GeneID', 'Symbol', 'Synonyms', 'description']

# Create a TextFileReader object
df_iterator = pd.read_csv(file_path, sep='\t', usecols=columns_to_use, chunksize=1000)

# Assume df_genes is already defined and contains the symbols we want to keep
# df_genes = pd.DataFrame({'Symbol': ['gene1', 'gene2', 'gene3']})

# Initialize an empty list to store the filtered chunks
filtered_chunks = []

# Iterate through all chunks
for chunk in df_iterator:
    # Filter the chunk to keep only rows where Symbol is in df_genes
    filtered_chunk = chunk[chunk['Symbol'].isin(df_genes['HUGO'])]
    
    # Append the filtered chunk to our list
    filtered_chunks.append(filtered_chunk)

# Concatenate all filtered chunks into a single DataFrame
result = pd.concat(filtered_chunks, ignore_index=True)

# Display the result
print(result)

# If you want to save this result to a new file:
# result.to_csv('filtered_output.csv', index=False)

# Print the total number of rows in the result
print(f"Total rows after filtering: {len(result)}")

result.to_parquet('test.pq')


# %%


# # File path
# file_path = 'gene_info'

# # Columns we want to retrieve
# columns_to_use = ['GeneID', 'Symbol', 'Synonyms', 'description']

# # Create a TextFileReader object
# # chunksize=1000 means we'll process 1000 rows at a time
# # usecols specifies which columns to read
# # sep='\t' indicates that the file is tab-delimited
# df_iterator = pd.read_csv(file_path, sep='\t', usecols=columns_to_use, chunksize=1000)

# # Get the first chunk (1000 rows)
# first_chunk = next(df_iterator)

# # drop all rows where the 'Symbol' column is 'NEWENTRY'
# first_chunk = first_chunk[first_chunk['Symbol'] != 'NEWENTRY']

# # Display the result
# print(first_chunk)

# # If you want to save this chunk to a new file:
# # first_chunk.to_csv('output.csv', index=False)

# # Note: The TextFileReader object (df_iterator) can be used to iterate 
# # over the entire file in chunks if needed in the future


# # %%
