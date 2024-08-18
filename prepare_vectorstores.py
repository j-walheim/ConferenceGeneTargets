# %%
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI
from utils.get_disease_terms import prepare_disease_synonyms
from utils.get_gene_synonyms import prepare_gene_synonyms
from tqdm import tqdm
from utils.vectorsstores_ontology import VectorStore

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
    

# # RAG function
# def rag(query):
#     gene_context = retrieve(gene_index, query)
#     disease_context = retrieve(disease_index, query, k = 2)
    
#     prompt = f"""Your task is to identify genes and diseases mentioned in the given text and map them to their standardized terms.
#     Use the following retrieved context to help identify genes, diseases, and their standard names:

#     Gene context:
#     {gene_context}

#     Disease context:
#     {disease_context}

#     If you identify a gene or disease mentioned in the text, provide its standardized term.
#     If you're unsure about a gene or disease mention, don't include it in the results.

#     Extract genes and diseases from the following text:

#     {query}
#     """
#     result = llm.invoke(prompt)
#     return result.content#(prompt)





# # Example usage
# result = rag("Recent studies have shown that mutations in the BRCA1 gene are associated with an increased risk of breast cancer.")
# print(result)
