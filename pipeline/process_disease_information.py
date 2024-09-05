import os
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from RAG_term_normalisation.vectorstore_ontology import VectorStore
from RAG_term_normalisation.get_disease_terms import prepare_disease_synonyms
from RAG_term_normalisation.get_gene_synonyms import prepare_gene_synonyms

load_dotenv()

# Prepare disease and gene synonyms
disease_synonyms = prepare_disease_synonyms()
gene_synonyms = prepare_gene_synonyms()

vectorstore = VectorStore()
vectorstore.prepare_lookups(gene_synonyms, disease_synonyms)
vectorstore.prepare_vectorstores()


class Abstract(BaseModel):
    disease: Optional[str] = Field(None, description="The primary disease or cancer type discussed in the abstract")

def extract_disease_info(abstract_text, model = 'groq'):

    if model == 'groq':
        llm = ChatGroq(
            model='llama-3.1-70b-versatile',
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
    else:
        llm = ChatMistralAI(
            model='mistral-large-latest',
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        
    disease_context = vectorstore.rag(abstract_text)

    print(disease_context)

    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting information from academic abstracts, with a focus on oncology. Your task is to accurately extract the primary disease or cancer type discussed in the abstract. Extract the information according to the following structure:

Disease: The primary disease or cancer type that is explicitly stated as the main focus of the research or treatment in the abstract. This should be a specific type of cancer or disease, not just a general category.If multiple diseases are mentioned, list all of them.

Follow these guidelines strictly:
- NEVER GUESS OR INFER INFORMATION THAT IS NOT EXPLICITLY STATED IN THE ABSTRACT.
- If no specific disease or cancer type is mentioned, or if it's a general study about cancer, use "Cancer (general)".
- Do not attempt to fill in missing information based on context or general knowledge.
- Accuracy is paramount. It is better to omit information than to provide potentially incorrect data.

Examples:
These data lay the foundation for first-in-human clinical translation of a T-cell based therapy targeting Ewing sarcoma. // Disease: Ewing sarcoma
We screened compounds for their ability to reduce CD8+ T cell senescence in various cancer types. // Disease: Cancer (general)
Our methodology allows to cultivate better CD8+ T cells. // Disease: None
This research investigates the efficacy of immunotherapy in triple-negative breast cancer (TNBC). // Disease: Triple-negative breast cancer
"""),
    ("human", "{text}")
])

    extraction_chain = prompt | llm.with_structured_output(schema=Abstract)
    result = extraction_chain.invoke({"text": abstract_text})
    return result

# Example usage
if __name__ == "__main__":
    abstract_text = "Your abstract text here..."
    result = extract_disease_info(abstract_text)
    print(f"Extracted disease: {result.disease}")