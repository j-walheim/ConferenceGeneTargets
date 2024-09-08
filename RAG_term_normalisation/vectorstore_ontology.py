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

    def prepare_lookup(self, data, name, synonym_col, official_col, description_col=None):
        if description_col:
            self.lookups[name] = [
                (synonym, row[official_col], row[description_col])
                for _, row in data.iterrows()
                for synonym in row[synonym_col].split(';') if row[synonym_col] != '-'
            ]
        else:
            self.lookups[name] = [
                (synonym, row[official_col])
                for _, row in data.iterrows()
                for synonym in row[synonym_col].split(';') if row[synonym_col] != '-'
            ]

    def create_or_load_index(self, name):
        index = faiss.IndexFlatIP(self.dimension)
        vectorstore_path = os.path.join(self.vectorstore_dir, f"vectorstore_{name}.index")
        if os.path.exists(vectorstore_path):
            print(f"Loading existing {name} vectorstore from {vectorstore_path}")
            index = faiss.read_index(vectorstore_path)
        else:
            print(f"Creating new {name} vectorstore at {vectorstore_path}")
            for item in tqdm(self.lookups[name], desc=f"Adding to {name} index"):
                synonym = item[0]
                text_to_encode = synonym
                if len(item) > 2 and item[2]:  # Check if third column exists and is not empty
                    text_to_encode += f" {item[2]}"
                vector = self.model.encode([text_to_encode])
                faiss.normalize_L2(vector)
                index.add(vector)
            faiss.write_index(index, vectorstore_path)
        self.indexes[name] = index

    def prepare_vectorstores(self):
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        for name in self.lookups:
            self.create_or_load_index(name)

    def retrieve(self, name, query):
        # Normalize the query vector
        query_vector = self.model.encode([query])
        faiss.normalize_L2(query_vector)
        # Ensure query_vector is a 2D numpy array
        query_vector = query_vector.reshape(1, -1)
        
        D, I = self.indexes[name].search(query_vector, k=5)
        logging.info(f"Query: {query}, Search results: D={D}, I={I}")
        
        results = [self.lookups[name][i][0] for i in I[0] if i < len(self.lookups[name])]
        return ", ".join(results) if results else ""

    def rag(self, name, query):
        return self.retrieve(name, query)


