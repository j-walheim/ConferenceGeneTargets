import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

class VectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2', vectorstore_dir="data/RAG_LLM/vectorstore"):
        self.model = SentenceTransformer(model_name)
        self.vectorstore_dir = vectorstore_dir
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.indexes = {}
        self.lookups = {}

    def prepare_lookup(self, data, name, synonym_col, official_col):
        self.lookups[name] = [
            (synonym, row[official_col])
            for _, row in data.iterrows()
            for synonym in row[synonym_col].split(';') if row[synonym_col] != '-'
        ]

    def create_or_load_index(self, name):
        index = faiss.IndexFlatL2(self.dimension)
        vectorstore_path = os.path.join(self.vectorstore_dir, f"vectorstore_{name}.index")
        if os.path.exists(vectorstore_path):
            print(f"Loading existing {name} vectorstore from {vectorstore_path}")
            index = faiss.read_index(vectorstore_path)
        else:
            print(f"Creating new {name} vectorstore at {vectorstore_path}")
            for synonym, _ in tqdm(self.lookups[name], desc=f"Adding to {name} index"):
                index.add(np.array(self.model.encode([synonym])))
            faiss.write_index(index, vectorstore_path)
        self.indexes[name] = index

    def prepare_vectorstores(self):
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        for name in self.lookups:
            self.create_or_load_index(name)

    def retrieve(self, name, query):
        D, I = self.indexes[name].search(np.array(self.model.encode([query])), k=5)
        logging.info(f"Query: {query}, Search results: D={D}, I={I}")
        
        results = [self.lookups[name][i] for i in I[0] if i < len(self.lookups[name])]
        return ", ".join(results) if results else ""

    def rag(self, name, query):
        return self.retrieve(name, query)
