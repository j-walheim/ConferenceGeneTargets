# %%
import requests
 
from biomart import BiomartServer
import pandas as pd
from dagster import asset
import os
import gzip
import shutil
import requests
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
 
 
 
def get_synonyms(df_hugo):

    
    
    # download full ncbi gene info
    url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz"

    # Local filenames
    gz_file = "data/RAG_LLM/features_raw/gene_info.gz"
    extracted_file = "data/RAG_LLM/features_raw/gene_info"

    if not os.path.exists(extracted_file):
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
    else:
        print(f"File {extracted_file} already exists. Skipping download and extraction.")



    # File path
    file_path = 'data/RAG_LLM/features_raw/gene_info'
    
    # Columns we want to retrieve
    columns_to_use = ['GeneID', 'Symbol', 'Synonyms', 'description']

    # Create a TextFileReader object
    df_iterator = pd.read_csv(file_path, sep='\t', usecols=columns_to_use, chunksize=1000)

    # Initialize an empty list to store the filtered chunks
    filtered_chunks = []

    # Iterate through all chunks
    from tqdm import tqdm

    for chunk in tqdm(df_iterator, desc="Processing chunks"):
        # Filter the chunk to keep only rows where Symbol is in df_genes
        filtered_chunk = chunk[chunk['Symbol'].isin(df_hugo['HUGO'])]
        
        # Append the filtered chunk to our list
        filtered_chunks.append(filtered_chunk)

    # Concatenate all filtered chunks into a single DataFrame
    result = pd.concat(filtered_chunks, ignore_index=True)

    # Display the result
    print(result)

    # If you want to save this result to a new file:
    result.to_csv('filtered_output.csv', index=False)

    # Print the total number of rows in the result
    print(f"Total rows after filtering: {len(result)}")

    return result

def prepare_gene_synonyms():
    fname_out = 'data/RAG_LLM/features/genes_synonyms.pq'
    
    if not os.path.exists(fname_out):
        df_genes = get_hugo_symbols_df()
        df_synonyms = get_synonyms(df_genes)
    
        df_synonyms.to_parquet(fname_out)
    else:
        df_synonyms = pd.read_parquet(fname_out)
    
    # Group by Symbol and aggregate the other columns
    genes_aggregated = df_synonyms.groupby('Symbol').agg({
        'Synonyms': lambda x: ';'.join(set(y for y in ';'.join(x).split(';') if y != '-') or ''),
        'description': lambda x: '; '.join(set(x))
    }).reset_index()

    # Clean up any potential extra spaces in the description
    genes_aggregated['description'] = genes_aggregated['description'].str.strip()

    # Add identity mapping for all Hugo symbols
    genes_aggregated['Synonyms'] = genes_aggregated.apply(lambda row: f"{row['Symbol']};{row['Synonyms']}" if row['Synonyms'] else row['Symbol'], axis=1)

    # Sort the synonyms alphabetically within each group
    genes_aggregated['Synonyms'] = genes_aggregated['Synonyms'].apply(lambda x: ';'.join(sorted(set(x.split(';')))))

    # Drop duplicates
    genes_aggregated = genes_aggregated.drop_duplicates()

    # Save the result
    genes_aggregated.to_csv('data/RAG_LLM/features/genes_synonyms_grouped.csv', index=False)
    return genes_aggregated
