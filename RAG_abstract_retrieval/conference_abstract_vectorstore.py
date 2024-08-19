import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class ConferenceAbstractVectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2', vectorstore_dir="/teamspace/studios/this_studio/ConferenceGeneTargets/data/production/abstract_vectorstore"):
        self.model = SentenceTransformer(model_name)
        self.vectorstore_dir = vectorstore_dir
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.abstract_lookup = []

    def create_or_load_index(self):
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        vectorstore_path = os.path.join(self.vectorstore_dir, "abstract_vectorstore.index")
        
        if os.path.exists(vectorstore_path):
            print("Loading existing index...")
            self.index = faiss.read_index(vectorstore_path)
        else:
            print("Creating new index...")
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.load_abstract_lookup()
        print(f"Index size: {self.index.ntotal}")

    def add_to_index(self, abstracts):
        embeddings = []
        for abstract in abstracts:
            embedding = self.model.encode(abstract['content'])
            embeddings.append(embedding)
            self.abstract_lookup.append(abstract)
        
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        self.save_abstract_lookup()

    def incremental_update(self, new_abstract_files):
        processed_files = self.get_processed_files()
        new_abstracts = []
        
        for file in new_abstract_files:
            if file not in processed_files:
                with open(file, 'r') as f:
                    abstract_data = json.load(f)
                    new_abstracts.extend(abstract_data)
                self.add_processed_file(file)
        
        if new_abstracts:
            self.add_to_index(new_abstracts)
            faiss.write_index(self.index, os.path.join(self.vectorstore_dir, "abstract_vectorstore.index"))
            print(f"Added {len(new_abstracts)} new abstracts to the index.")
        else:
            print("No new abstracts to add. All files have been processed.")

    def retrieve(self, query, k=5):
        query_vector = self.model.encode([query])
        D, I = self.index.search(query_vector, k)
        return [self.abstract_lookup[i] for i in I[0]]

    def add_processed_file(self, filename):
        processed_files_path = os.path.join(self.vectorstore_dir, "processed_files.txt")
        with open(processed_files_path, "a") as f:
            f.write(filename + "\n")

    def get_processed_files(self):
        processed_files_path = os.path.join(self.vectorstore_dir, "processed_files.txt")
        if os.path.exists(processed_files_path):
            with open(processed_files_path, "r") as f:
                return set(f.read().splitlines())
        return set()

    def save_abstract_lookup(self):
        lookup_path = os.path.join(self.vectorstore_dir, "abstract_lookup.json")
        with open(lookup_path, "w") as f:
            json.dump(self.abstract_lookup, f)

    def load_abstract_lookup(self):
        lookup_path = os.path.join(self.vectorstore_dir, "abstract_lookup.json")
        if os.path.exists(lookup_path):
            with open(lookup_path, "r") as f:
                self.abstract_lookup = json.load(f)
