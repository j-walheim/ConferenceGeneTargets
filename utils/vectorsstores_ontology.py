import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class VectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2', vectorstore_dir="data/RAG_LLM/vectorstore"):
        self.model = SentenceTransformer(model_name)
        self.vectorstore_dir = vectorstore_dir
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.gene_index = None
        self.disease_index = None
        self.gene_lookup = None
        self.disease_lookup = None

    def prepare_lookups(self, gene_synonyms, disease_synonyms):
        self.gene_lookup = [
            f"HUGO: {row['Symbol']}{', Synonyms: ' + row['Synonyms'] if row['Synonyms'] != '-' else ''}"
            for _, row in gene_synonyms.iterrows()
        ]
        self.disease_lookup = [
            f"Disease: {row['disease']}, Synonyms: {row['synonyms']}"
            for _, row in disease_synonyms.iterrows()
        ]

    def add_to_index(self, index, texts):
        for text in tqdm(texts, desc="Adding to index"):
            embedding = self.model.encode([text])
            index.add(np.array(embedding))

    def create_or_load_index(self, name, lookup):
        index = faiss.IndexFlatL2(self.dimension)
        vectorstore_path = os.path.join(self.vectorstore_dir, f"vectorstore_{name}.index")
        if os.path.exists(vectorstore_path):
            print(f"Loading existing {name} vectorstore from {vectorstore_path}")
            index = faiss.read_index(vectorstore_path)
        else:
            print(f"Creating new {name} vectorstore at {vectorstore_path}")
            self.add_to_index(index, lookup)
            faiss.write_index(index, vectorstore_path)
        return index

    def prepare_vectorstores(self):
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        self.gene_index = self.create_or_load_index("genes", self.gene_lookup)
        self.disease_index = self.create_or_load_index("diseases", self.disease_lookup)

    def retrieve(self, index, query, k=5):
        query_vector = self.model.encode([query])
        D, I = index.search(query_vector, k)
        return [self.gene_lookup[i] if index is self.gene_index else self.disease_lookup[i] for i in I[0]]

    def rag(self, query):
        gene_context = self.retrieve(self.gene_index, query)
        disease_context = self.retrieve(self.disease_index, query, k=2)
        return gene_context, disease_context
