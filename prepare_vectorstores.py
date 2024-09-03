# %%
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI
from RAG_term_normalisation.get_disease_terms import prepare_disease_synonyms
from RAG_term_normalisation.get_gene_synonyms import prepare_gene_synonyms
from tqdm import tqdm
from RAG_term_normalisation.vectorstore_ontology import VectorStore

from dotenv import load_dotenv
load_dotenv()


# Prepare disease and gene synonyms
disease_synonyms = prepare_disease_synonyms()
gene_synonyms = prepare_gene_synonyms()

vectorstore = VectorStore()
vectorstore.prepare_lookups(gene_synonyms, disease_synonyms)
vectorstore.prepare_vectorstores()

# To use the RAG functionality
gene_context, disease_context = vectorstore.rag("Recent studies have shown that mutations in the BRCA1 gene are associated with an increased risk of breast cancer.")

os.makedirs('tmp', exist_ok=True)
# write results to text
with open('tmp/results.txt', 'w') as f:
    f.write(f"Gene context: {gene_context}\n")
    f.write(f"Disease context: {disease_context}\n")
    
