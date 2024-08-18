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
from .config import NUM_PARTITIONS, DEV_PAGE_LIMIT, STORAGE_DIR, ENVIRONMENT
import json
from typing import List
from dotenv import load_dotenv
load_dotenv()

from pipeline.helpers import log_progress

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")




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
@task
def download_pdf_task(**kwargs):
    ti = kwargs['ti']
    
    log_progress(ti, f"Downloading PDF... to {STORAGE_DIR}")
    pdf_url = 'https://www.aacr.org/wp-content/uploads/2024/04/AACR2024_Regular_Abstracts_04-01-24.pdf'
    pdf_path = os.path.join(STORAGE_DIR, 'pdfs')
    
    os.makedirs(pdf_path, exist_ok=True)
    fname = os.path.join(pdf_path, os.path.basename(pdf_url))
    
    log_progress(ti, f"PDF name {fname}")
    
    if not os.path.exists(fname):
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open(fname, 'wb') as file:
            file.write(response.content)
        log_progress(ti, f"PDF downloaded to {fname}")
    else:
        log_progress(ti, "PDF already downloaded.")
    
    return fname

@task
def partition_into_batches(pdf_file, batch_size: int = 100, **kwargs) -> List[List[int]]:
    ti = kwargs['ti']

    reader = PdfReader(pdf_file)
    total_pages = len(reader.pages)
    
    log_progress(ti, f"Partitioning {total_pages} PDF pages into batches of size {batch_size}")
    
    
    if ENVIRONMENT == 'development':
        total_pages = DEV_PAGE_LIMIT 
        log_progress(ti, f"Development mode: Processing only the first {DEV_PAGE_LIMIT} pages")
    return [list(range(i, min(i + batch_size, total_pages))) for i in range(0, total_pages, batch_size)]



@task
def process_pdf_partition(batch_pages: List[int], **kwargs):
    ti = kwargs['ti']
    pdf_file = '/teamspace/studios/this_studio/ConferenceGeneTargets/data/pdfs/AACR2024_Regular_Abstracts_04-01-24.pdf'
    pages_dir = os.path.join(STORAGE_DIR, ENVIRONMENT, 'processed_pages')    
    os.makedirs(pages_dir, exist_ok=True)

    log_progress(ti, f"Processing PDF partition {batch_pages[0]} to {batch_pages[-1]}")

    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown"
    )

    reader = PdfReader(pdf_file)
    total_pages = len(reader.pages)
    
    if ENVIRONMENT == 'development':
        total_pages = min(total_pages, DEV_PAGE_LIMIT)
        log_progress(ti, f"Development mode: Processing only the first {DEV_PAGE_LIMIT} pages")
    

    processed_files = []
    for i in batch_pages:#range(start_page, end_page):
        output_file = os.path.join(pages_dir, f'page_{i+1}.json')
        
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            log_progress(ti, f"Skipping page {i+1} as it has already been processed")
            processed_files.append(output_file)
            continue
        
        page_docs = process_page(reader.pages[i], i + 1, parser)
        if not page_docs:
            raise AirflowException(f"No documents extracted from page {i+1}, most likely this means that the quota is exceeded")
        
        if not page_docs is None:
            with open(output_file, 'w') as f:
                json.dump([{'page_number': doc.metadata['page_number'], 'content': doc.text} for doc in page_docs], f)
      #  with open(output_file, 'w') as f:
      #      json.dump([{'page_number': doc.metadata['page_number'], 'content': doc.text} for doc in page_docs], f)
        
        processed_files.append(output_file)
        log_progress(ti, f"Processed and saved page {i+1} of partition")

    log_progress(ti, f"Finished processing PDF partition, saved {len(processed_files)} page files.")

    return pages_dir


@task
def combine_pdf_partitions(**kwargs):
    ti = kwargs['ti']
    log_progress(ti, "Starting to combine PDF partitions.")
    
    partition_files = [ti.xcom_pull(task_ids=f'process_partitions.process_partition_{i}') for i in range(1, NUM_PARTITIONS + 1)]
    valid_files = [file for file in partition_files if file and os.path.exists(file)]
    log_progress(ti, f"Found {len(valid_files)} valid partition files.")
    
    if not valid_files:
        raise AirflowException("No valid partition files found to combine.")
    
    dfs = [pd.read_pickle(file) for file in valid_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    
    output_file = os.path.join(STORAGE_DIR, f'combined_pdf_content_{ENVIRONMENT}.pkl')
    os.makedirs(STORAGE_DIR, exist_ok=True)
    combined_df.to_pickle(output_file)
    
    log_progress(ti, f"Finished combining PDF partitions for {ENVIRONMENT} environment")
    return output_file