# %% 
import os
import sys
from typing import List
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
import pickle
from defs.abstract_class import Abstract
import json
from airflow.decorators import task
import pandas as pd
import glob
from .config import NUM_PARTITIONS, DEV_PAGE_LIMIT, STORAGE_DIR, ENVIRONMENT
from pydantic import BaseModel, Field
from typing import List, Optional

sys.path.append('/teamspace/studios/this_studio/ConferenceGeneTargets')

from RAG.vectorstore_ontology import VectorStore
from RAG.get_disease_terms import prepare_disease_synonyms
from RAG.get_gene_synonyms import prepare_gene_synonyms
from pipeline.helpers import log_progress
from dotenv import load_dotenv
load_dotenv()


# Prepare disease and gene synonyms
disease_synonyms = prepare_disease_synonyms()
gene_synonyms = prepare_gene_synonyms()

vectorstore = VectorStore()
vectorstore.prepare_lookups(gene_synonyms, disease_synonyms)
vectorstore.prepare_vectorstores()

def extract_abstract_info(abstract_text, model = 'mistral'):
    
    if model == 'mistral':
        llm = ChatMistralAI(
        model='open-mistral-nemo',#'mistral-large-latest',
        temperature=0 )
    else:
        llm = ChatGroq(
            model='llama3-groq-70b-8192-tool-use-preview',
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )


    gene_context, disease_context = vectorstore.rag(abstract_text)
    
    print(gene_context, disease_context)
    
    prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are an expert at extracting information from academic abstracts, with a focus on genetics and oncology. Your task is to accurately and completely extract the requested information, paying close attention to details. Extract the information according to the following structure:

1. abstract_number: Extract ONLY the 4-digit number from the abstract number. Ignore any preceding or following characters. For example, if you see '0009AFNT-212', extract ONLY '0009'. If no 4-digit number is present, use 'n/a'.

2. title: The title of the abstract, preserving capitalization and any special characters exactly as they appear. Make sure to only provide the title, not the full text.

3. authors: A comprehensive list of all authors mentioned, each containing:
   - name: The full name of the author, exactly as it appears in the abstract
   - affiliation: The institution or organization the author is affiliated with. If not available, use null.

4. disease: A list of specific cancer indications or types mentioned in the abstract. Use ONLY the terms provided in the following disease context: {disease_context}. Map synonyms to the primary term. If it is about cancer but not a specific one, put 'Cancer general'. If a cancer type is mentioned that's not in this list, do not include it. If no valid cancer types are mentioned, return an empty list.

5. gene: A list of genes mentioned in the abstract, using only the gene symbols provided in the following gene context: {gene_context}. Map synonyms to the primary term. Include only genes directly related to the discussed cancer or research topic. If no genes from this list are mentioned, return an empty list.

6. organism: The type of organism for which the evidence is provided. Can be 'cell line', 'PDX', 'animal', 'human', or 'n/a' if not specified. Use 'animal' if any of the following are mentioned: mouse, mice, rat, rabbit, guinea pig, hamster, dog, pig, monkey, or primate. If multiple organism types are used, list all that apply.

7. trial_stage: The stage of the trial or study. Can be 'preclinical', 'Phase I', 'Phase II', 'Phase III', 'post-approval', or 'n/a' if not applicable or not specified.

8. compound_name: The name of the compound or drug mentioned in the abstract. E.g. 'GS-P-328', 'IOMX-0675', or 'Pembrolizumab'

9. abstract_category: Assign the abstract to one of the following categories: 'Gene-Disease Associations' if it is about the impact of some genes on the disease, 'Diagnostics' if it is about methods for diagnosis and does not provide a novel finding on a gene, or 'Other'. Choose the most appropriate category based on the main focus of the abstract.

Follow these guidelines strictly:
- For the abstract number, extract ONLY the 4-digit number. Ignore any other characters.
- Include all authors mentioned in the abstract, even if their affiliation is not provided.
- If any information is not available or cannot be determined with certainty, use null for optional fields or an empty list for list fields.
- Never guess or infer information that is not explicitly stated in the abstract.
- Do not attempt to fill in missing information based on context or general knowledge.

Accuracy is paramount. It is better to omit information than to provide potentially incorrect data. Under no circumstances should you guess or make assumptions about any data."""),
    ("human", "{text}")
])

    
    # Create the extraction chain
    extraction_chain = prompt | llm.with_structured_output(schema=Abstract)
    result = extraction_chain.invoke({"text": abstract_text})
    return result
 


@task
def process_abstracts_partition(batch_pages: List[int], model = 'groq', **kwargs):
    ti = kwargs['ti']
    pages_processed_dir = os.path.join(STORAGE_DIR, ENVIRONMENT, 'processed_pages')    
    pages_parsed_dir = os.path.join(STORAGE_DIR, ENVIRONMENT, f'parsed_pages_{model}')    
    os.makedirs(pages_parsed_dir, exist_ok=True)   
    
    for i in batch_pages:
        
        input_file = os.path.join(pages_processed_dir, f'page_{i+1}.json')
        output_file = os.path.join(pages_parsed_dir, f'page_{i+1}.json')
        
        if os.path.exists(output_file):
            log_progress(ti,f"Skipping page {i+1} as it has already been processed.")
            continue
        
        log_progress(ti,f"Processing page {i+1} with model {model}...")
        with open(input_file) as f:
            abstract = json.load(f)
        abstract_text = abstract[0].get('content')
        
        # %%
        result = extract_abstract_info(abstract_text, model='groq')
        abstract_dict = result.dict()
        abstract_dict['text'] = abstract_text
        abstract_dict['page_number'] = abstract[0].get('page_number')
        
        json_result = json.dumps(abstract_dict, indent=2)        
        
        with open(output_file, 'w') as f:
            f.write(json_result)
        

#@task
def create_jsonl_from_parsed_pages(**kwargs):
#    ti = kwargs['ti']
    model = 'groq'
    parsed_pages_dir = os.path.join(STORAGE_DIR, ENVIRONMENT, f'parsed_pages_{model}') 
    output_jsonl = os.path.join(STORAGE_DIR, ENVIRONMENT, f'parsed_pages_{model}.jsonl')
    with open(output_jsonl, 'w') as jsonl_file:
        for filename in sorted(glob.glob(os.path.join(parsed_pages_dir, 'page_*.json'))):
            with open(filename, 'r') as json_file:
                abstract_dict = json.load(json_file)
                json.dump(abstract_dict, jsonl_file)
                jsonl_file.write('\n')
    
 #   log_progress(ti, f"Created JSONL file: {output_jsonl}")

@task
def process_and_merge_abstracts(processed_partition_files):
    processed_partitions = []
    for processed_partition_file in processed_partition_files:
        with open(processed_partition_file, 'r') as f:
            processed_abstracts = json.load(f)
        processed_partitions.extend(processed_abstracts)
    
    output_file = os.path.join(STORAGE_DIR,ENVIRONMENT, 'merged_processed_abstracts.json')
    with open(output_file, 'w') as f:
        json.dump(processed_partitions, f, indent=2)
    
    return output_file