import sys
sys.path.append('/teamspace/studios/this_studio/ConferenceGeneTargets')
from RAG_abstract_retrieval.conference_abstract_vectorstore import ConferenceAbstractVectorStore

class ConferenceAbstractRAG:
    def __init__(self):
        self.vectorstore = ConferenceAbstractVectorStore()

    def initialize(self):
        self.vectorstore.create_or_load_index()

    def update(self, new_abstract_files):
        self.vectorstore.incremental_update(new_abstract_files)

    def answer_query(self, query):
        relevant_abstracts = self.vectorstore.retrieve(query, k=5)
        return self.format_results(query, relevant_abstracts)

    def format_results(self, query, relevant_abstracts):
        result = f"Query: {query}\n\nRelevant Abstracts:\n"
        for i, abstract in enumerate(relevant_abstracts, 1):
            result += f"\n{i}. Page {abstract['page_number']}:\n{abstract['content'][:300]}...\n"
        return result

if __name__ == "__main__":
    rag_system = ConferenceAbstractRAG()
    rag_system.initialize()
    
    # Example usage
    import os
    import glob

    data_dir = '/teamspace/studios/this_studio/ConferenceGeneTargets/data/production/processed_pages'
    new_abstract_files = glob.glob(os.path.join(data_dir, '*.json'))    
    rag_system.update(new_abstract_files)
    
    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        result = rag_system.answer_query(query)
        print(result)
