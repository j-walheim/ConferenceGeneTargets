import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
from pinecone import Pinecone, ServerlessSpec
import time 
import pandas as pd
import requests 
from dotenv import load_dotenv

load_dotenv()

class VectorStore_genes:
    def __init__(self):
        
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    def  prepareGenes(self):

        # Download the file
        url = "https://g-a8b222.dd271.03c0.data.globus.org/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt"
        response = requests.get(url)
        with open("hgnc_complete_set.txt", "wb") as file:
            file.write(response.content)

        # Load as TSV
        df = pd.read_csv("hgnc_complete_set.txt", sep="\t")

        # Select relevant columns
        relevant_columns = ['symbol', 'alias_symbol', 'prev_symbol', 'name', 'alias_name', 'prev_name']
        df_selected = df[relevant_columns]

        # Function to split and explode multiple values in a cell
        def split_and_explode(df, column):
            return df.assign(**{column: df[column].str.split('|')}).explode(column)

        # Convert from wide to long format
        df_long = pd.melt(df_selected, id_vars=['symbol'], var_name='synonym_type', value_name='synonym')

        # Split and explode multiple synonyms
        df_long = split_and_explode(df_long, 'synonym')

        # Remove rows with NaN synonyms
        df_long = df_long.dropna(subset=['synonym'])

        # Remove duplicate rows
        df_long = df_long.drop_duplicates()

        # Reset index
        df_long = df_long.reset_index(drop=True)

        # Save to CSV
        df_long.to_csv('gene_synonyms.csv', index=False)

        df_long = df_long.drop(columns=['synonym_type'])

        # add identity
        df_identity = df_long.copy()
        df_identity['synonym'] = df_identity['symbol']
        df_identity = df_identity.drop_duplicates()

        df_long = pd.concat([df_long, df_identity], ignore_index=True)

        df_long = df_long.drop_duplicates().reset_index(drop=True)

        print(df_long.head(20))
        print(f"Total number of synonym entries: {len(df_long)}")



        return df_long
    
    def create_or_load_index(self, index_name = 'gene-index'):
        
        
        # Check if the index exists
        index_names = [index['name'] for index in self.pc.list_indexes()]
        if index_name in index_names:
            print(f"Index {index_name} already exists.")
            return
        else:        

        
            self.pc.create_index(
                name=index_name,
                dimension=1024, # Replace with your model dimensions
                metric="cosine", # Replace with your model metric
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ) 
            )
            # Wait for the index to be ready
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        
        df_genes = self.prepareGenes()
  

        batch_size = 96

        
        for i in tqdm(range(0, len(df_genes), batch_size), desc="Processing batches"):
            batch = df_genes.iloc[i:i+batch_size]
            
            embeddings = self.pc.inference.embed(
                model="multilingual-e5-large",
                inputs=batch['synonym'].tolist(),
                parameters={"input_type": "passage", "truncate": "END"}
            )
            
            
            vectors = []
            for d, e in zip(batch.itertuples(), embeddings.data):
                vectors.append({
                    "id": str(d.Index),
                    "values": e['values'],
                    "metadata": {'synonym': d.synonym, "symbol": d.symbol,}
                })
        
            index = self.pc.Index(index_name)
            index.upsert(
                vectors=vectors,
                namespace="ns1"
            )
            
            time.sleep(1)
            
    def retrieve(self, index_name, query):
        max_retries = 5
        base_wait_time = 20

        for attempt in range(max_retries):
            try:
                index = self.pc.Index(index_name)

                embedding = self.pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=[query],
                    parameters={
                        "input_type": "query"
                    }
                )

                results = index.query(
                    namespace="ns1",
                    vector=embedding[0].values,
                    top_k=10,
                    include_values=False,
                    include_metadata=True
                )

                metadata = [d['metadata'] for d in results['matches']]
                return metadata

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = base_wait_time * (2 ** attempt)
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Returning empty list.")
                    return []

        return []  # should never be reached


