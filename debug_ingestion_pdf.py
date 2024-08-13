import os
import requests
from llama_parse import LlamaParse
import pickle
from PyPDF2 import PdfReader, PdfWriter
import logging
import pandas as pd
from airflow.models import Variable
from airflow.exceptions import AirflowException
from airflow.decorators import task
from colorama import init, Fore, Back, Style



from pipeline.config import NUM_PARTITIONS, DEV_PAGE_LIMIT

from dotenv import load_dotenv
load_dotenv()

# %% 
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from defs.abstract_class import Author, GeneDisease, Abstract

import os
import getpass
import pickle

from dotenv import load_dotenv
load_dotenv()




# Configuration
ENVIRONMENT = os.getenv("environment", "development")
STORAGE_DIR = os.getenv("STORAGE_DIR")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")


def log_progress(ti, message):
    ti.xcom_push(key='progress_log', value=message)
    print(Back.GREEN + message + Style.RESET_ALL)

def process_page(page_content, page_number, parser):
    temp_file = f"temp_page_{page_number}.pdf"
    writer = PdfWriter()
    writer.add_page(page_content)
    with open(temp_file, "wb") as output_file:
        writer.write(output_file)
    
    documents = parser.load_data(temp_file)
    os.remove(temp_file)
    
    for doc in documents:
        doc.metadata['page_number'] = page_number
    
    return documents



partition_number = 0
pages_dir = os.path.join(STORAGE_DIR, ENVIRONMENT, 'processed_pages')    
os.makedirs(pages_dir, exist_ok=True)
output_file = os.path.join(pages_dir, f'partition_{partition_number}.csv')
pdf_file = 'data/pdfs/AACR2024_Regular_Abstracts_04-01-24.pdf'


#log_progress(ti, f"Processing PDF partition {partition_number}")

parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    result_type="markdown"
)

reader = PdfReader(pdf_file)
total_pages = len(reader.pages)

if ENVIRONMENT == 'development':
    total_pages = min(total_pages, DEV_PAGE_LIMIT)
  #  log_progress(ti, f"Development mode: Processing only the first {DEV_PAGE_LIMIT} pages")

pages_per_partition = max(1, total_pages // NUM_PARTITIONS)
start_page = partition_number * pages_per_partition
end_page = min((partition_number + 1) * pages_per_partition, total_pages)

all_documents = []
for i in range(start_page, end_page):
    page_docs = process_page(reader.pages[i], i + 1, parser)
    all_documents.extend(page_docs)
    print(f"Processed page {i+1}/{end_page} of partition {partition_number}")

df = pd.DataFrame({
    'page_number': [doc.metadata['page_number'] for doc in all_documents],
    'content': [doc.text for doc in all_documents]
})
df.to_csv(output_file, index=False)

#log_progress(ti, f"Finished processing PDF partition {partition_number}, extracted {len(all_documents)} documents.")

#return output_file


# 

abstract = pickle.load(open('/teamspace/studios/this_studio/ConferenceGeneTargets/data/development/processed_pages/partition_0.pkl'))
