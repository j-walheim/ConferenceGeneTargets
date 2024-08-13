# %% 
from typing import List
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
import pickle
from defs.abstract_class import Abstract, tcga_to_fullname
import json
from airflow.decorators import task
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from typing import List, Optional
ENVIRONMENT = os.getenv("environment", "development")
STORAGE_DIR = os.getenv("STORAGE_DIR")

def extract_abstract_info_mistral(abstract_row):
    # Initialize the Mistral AI model
    llm = ChatMistralAI(
        model='open-mistral-nemo',#'mistral-large-latest',
        temperature=0
    )
    
    
    # def extract_abstract_info(abstract_row):
    # # Initialize the Groq AI model
    # llm = ChatGroq(
    #     model='llama3-groq-70b-8192-tool-use-preview',
    #     temperature=0,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2
    # )
    
    valid_full_names = list(tcga_to_fullname.values())
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert at extracting information from academic abstracts, with a focus on genetics and oncology. Your task is to accurately and completely extract the requested information, paying close attention to details. Extract the information according to the following structure:

1. abstract_number: Extract ONLY the 4-digit number from the abstract number. Ignore any preceding or following characters. For example, if you see '0009AFNT-212', extract ONLY '0009'. If no 4-digit number is present, use 'n/a'.

2. title: The complete title of the abstract, preserving capitalization and any special characters exactly as they appear.

3. authors: A comprehensive list of all authors mentioned, each containing:
   - name: The full name of the author, exactly as it appears in the abstract
   - affiliation: The institution or organization the author is affiliated with. If not available, use null.

4. disease: A list of specific cancer indications or types mentioned in the abstract. Use ONLY the following full names: {', '.join(valid_full_names)}. If a cancer type is mentioned that's not in this list, do not include it. If no valid cancer types are mentioned, return an empty list.

5. gene: A list of genes mentioned in the abstract, using only official HUGO gene symbols. Include only genes directly related to the discussed cancer or research topic. If no genes are mentioned, return an empty list.

Follow these guidelines strictly:
- For the abstract number, extract ONLY the 4-digit number. Ignore any other characters.
- For genes, use only official HUGO gene symbols. If a gene is mentioned using a different nomenclature, convert it to the HUGO symbol if you are certain of the correct conversion. If unsure, do not include the gene.
- For cancer indications, use ONLY the full names provided in the list above. Do not include any cancer types not in this list.
- Preserve all formatting, capitalization, and special characters in the title and abstract text exactly as they appear in the original.
- Include all authors mentioned in the abstract, even if their affiliation is not provided.
- If any information is not available or cannot be determined with certainty, use null for optional fields or an empty list for list fields.
- Never guess or infer information that is not explicitly stated in the abstract.
- Do not attempt to fill in missing information based on context or general knowledge.

Accuracy is paramount. It is better to omit information than to provide potentially incorrect data. Under no circumstances should you guess or make assumptions about any data."""),
        ("human", "{text}")
    ])
    
    # Create the extraction chain
    extraction_chain = prompt | llm.with_structured_output(schema=Abstract)
    result = extraction_chain.invoke({"text": abstract_row['content']})
    abstract_dict = result.dict()
    abstract_dict['text'] = abstract_row['content']
    json_result = json.dumps(abstract_dict, indent=2)
    return json_result
 
def extract_abstract_info(abstract_row):
    # Initialize the Groq AI model
    llm = ChatGroq(
        model='llama3-groq-70b-8192-tool-use-preview',
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    
    valid_full_names = list(tcga_to_fullname.values())
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert at extracting information from academic abstracts, with a focus on genetics and oncology. Your task is to accurately and completely extract the requested information, paying close attention to details. Extract the information according to the following structure:

1. abstract_number: Extract ONLY the 4-digit number from the abstract number. Ignore any preceding or following characters. For example, if you see '0009AFNT-212', extract ONLY '0009'. If no 4-digit number is present, use 'n/a'.

2. title: The complete title of the abstract, preserving capitalization and any special characters exactly as they appear.

3. authors: A comprehensive list of all authors mentioned, each containing:
   - name: The full name of the author, exactly as it appears in the abstract
   - affiliation: The institution or organization the author is affiliated with. If not available, use null.

4. disease: A list of specific cancer indications or types mentioned in the abstract. Use ONLY the following full names: {', '.join(valid_full_names)}. If an equivalent or related cancer type is mentioned that's not in this list, map it to the most appropriate name from the list. If no valid cancer types or their equivalents are mentioned, return an empty list.

5. gene: A list of genes mentioned in the abstract, using only official HUGO gene symbols. Include only genes directly related to the discussed cancer or research topic. If no genes are mentioned, return an empty list.

Follow these guidelines strictly:
- For the abstract number, extract ONLY the 4-digit number. Ignore any other characters.
- For genes, use only official HUGO gene symbols. If a gene is mentioned using a different nomenclature, convert it to the HUGO symbol if you are certain of the correct conversion. If unsure, do not include the gene.
- For cancer indications, use ONLY the full names provided in the list above. If an equivalent or related cancer type is mentioned, map it to the most appropriate name from the list.
- Preserve all formatting, capitalization, and special characters in the title and abstract text exactly as they appear in the original.
- Include all authors mentioned in the abstract, even if their affiliation is not provided.
- If any information is not available or cannot be determined with certainty, use null for optional fields or an empty list for list fields.
- Never guess or infer information that is not explicitly stated in the abstract.
- Do not attempt to fill in missing information based on context or general knowledge.

Accuracy is paramount. It is better to omit information than to provide potentially incorrect data. Under no circumstances should you guess or make assumptions about any data."""),
        ("human", "{text}")
    ])
    
    # Create the extraction chain
    extraction_chain = prompt | llm.with_structured_output(schema=Abstract)
    result = extraction_chain.invoke({"text": abstract_row['content']})
    abstract_dict = result.dict()
    abstract_dict['text'] = abstract_row['content']
    json_result = json.dumps(abstract_dict, indent=2)
    return json_result


@task
def process_abstracts_partition(partition_file):
    df = pd.read_pickle(partition_file)
    processed_abstracts = []
    for _, row in df.iterrows():
        json_result = extract_abstract_info(row)
        processed_abstracts.append(json.loads(json_result))
    
    output_file = partition_file.replace('.pkl', '_processed.json')
    with open(output_file, 'w') as f:
        json.dump(processed_abstracts, f, indent=2)
    
    return output_file


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