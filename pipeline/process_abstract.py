# %% 
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import pickle
from defs.abstract_class import Author, GeneDisease, Abstract

import os
import getpass

from dotenv import load_dotenv
load_dotenv()

def extract_abstract_info(fname_abstract):
    
#    abstract = 
    
    # Initialize the Groq AI model
    llm = ChatGroq(
        model='llama3-groq-70b-8192-tool-use-preview',
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )


    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at extracting information from academic abstracts, with a focus on genetics and oncology. Your task is to accurately and completely extract the requested information, paying close attention to details. Extract the information according to the following structure:

        1. abstract_number: Extract ONLY the 4-digit number from the abstract number. Ignore any preceding or following characters. For example, if you see '0009AFNT-212', extract ONLY '0009'. If no 4-digit number is present, use 'n/a'.

        2. title: The full title of the abstract.

        3. authors: A list of authors, each containing:
        - name: The full name of the author
        - affiliation: The institution or organization the author is affiliated with. If not available, use 'n/a'.

        4. abstract_text: The full text of the abstract, with any missing whitespace corrected.

        5. gene_disease: A list of gene-disease interactions, each containing:
        - gene: The name of the gene (use only official HUGO gene symbols)
        - disease: The name of the disease (use OncoTree nomenclature for cancer indications)
        - description: The description of the interaction. If not available, use 'n/a'.
        - directionality: The directionality of the interaction. If not available, use 'n/a'.

        6. gene: A list of genes mentioned in the abstract (use only official HUGO gene symbols).

        Follow these guidelines strictly:
        - For the abstract number, extract ONLY the 4-digit number. Ignore any other characters.
        - For genes, use only official HUGO gene symbols. If a gene is mentioned using a different nomenclature, convert it to the HUGO symbol.
        - For cancer indications, use the OncoTree nomenclature. If the abstract uses a different nomenclature, convert it to the appropriate OncoTree term.
        - If any information is not available or cannot be determined with certainty, use 'n/a' (not applicable) for that field.
        - Never guess or infer information that is not explicitly stated in the abstract. If you are unsure about any information, use 'n/a'.
        - Do not attempt to fill in missing information based on context or general knowledge. If it's not in the abstract, it's 'n/a'.
        - Ensure the abstract text is complete and properly formatted with correct whitespace.

        Accuracy is paramount. It is better to use 'n/a' than to provide potentially incorrect information. Under no circumstances should you guess or make assumptions about any data."""),
        ("human", "{text}")
    ])

    # %% 
    # Create the extraction chain
    extraction_chain = prompt | llm.with_structured_output(schema=Abstract)