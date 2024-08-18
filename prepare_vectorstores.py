# %%
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI
from utils.get_disease_terms import prepare_disease_synonyms
from utils.get_gene_synonyms import prepare_gene_synonyms
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

# Prepare disease and gene synonyms
disease_synonyms = prepare_disease_synonyms()
gene_synonyms = prepare_gene_synonyms()

# %%
gene_lookup = [
    f"HUGO: {row['Symbol']}{', Synonyms: ' + row['Synonyms'] if row['Synonyms'] != '-' else ''}"
    for _, row in gene_synonyms.iterrows()
]

disease_lookup = [
    f"Disease: {row['disease']}, Synonyms: {row['synonyms']}"
    for _, row in disease_synonyms.iterrows()
]

# Set up embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create FAISS indexes
dimension = model.get_sentence_embedding_dimension()
gene_index = faiss.IndexFlatL2(dimension)
disease_index = faiss.IndexFlatL2(dimension)

# Function to add vectors to index
def add_to_index(index, texts):
    for text in tqdm(texts, desc="Adding genes to index"):
        embedding = model.encode([text])
        index.add(np.array(embedding))

# Define vectorstore directory
vectorstore_dir = "data/RAG_LLM/vectorstore"
if not os.path.exists(vectorstore_dir):
    os.makedirs(vectorstore_dir)

# Gene vectorstore
gene_vectorstore_path = os.path.join(vectorstore_dir, "vectorstore_genes.index")
if os.path.exists(gene_vectorstore_path):
    print(f"Loading existing gene vectorstore from {gene_vectorstore_path}")
    gene_index = faiss.read_index(gene_vectorstore_path)
else:
    print(f"Creating new gene vectorstore at {gene_vectorstore_path}")
    add_to_index(gene_index, gene_lookup)
    faiss.write_index(gene_index, gene_vectorstore_path)

# Disease vectorstore
disease_vectorstore_path = os.path.join(vectorstore_dir, "vectorstore_diseases.index")
if os.path.exists(disease_vectorstore_path):
    print(f"Loading existing disease vectorstore from {disease_vectorstore_path}")
    disease_index = faiss.read_index(disease_vectorstore_path)
else:
    print(f"Creating new disease vectorstore at {disease_vectorstore_path}")
    add_to_index(disease_index, disease_lookup)
    faiss.write_index(disease_index, disease_vectorstore_path)
    print("Disease vectorstore persisted successfully")

# Set up retrievers
def retrieve(index, query, k=5):
    query_vector = model.encode([query])
    D, I = index.search(query_vector, k)
    return [gene_lookup[i] if index is gene_index else disease_lookup[i] for i in I[0]]

# Set up LLM
llm = ChatMistralAI(model='open-mistral-nemo')



# RAG function
def rag(query):
    gene_context = retrieve(gene_index, query)
    disease_context = retrieve(disease_index, query, k = 2)
    
    prompt = f"""Your task is to identify genes and diseases mentioned in the given text and map them to their standardized terms.
    Use the following retrieved context to help identify genes, diseases, and their standard names:

    Gene context:
    {gene_context}

    Disease context:
    {disease_context}

    If you identify a gene or disease mentioned in the text, provide its standardized term.
    If you're unsure about a gene or disease mention, don't include it in the results.

    Extract genes and diseases from the following text:

    {query}
    """
    result = llm.invoke(prompt)
    return result.content#(prompt)





# Example usage
result = rag("Recent studies have shown that mutations in the BRCA1 gene are associated with an increased risk of breast cancer.")
print(result)
