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
from .config import NUM_PARTITIONS, DEV_PAGE_LIMIT

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
def process_pdf_partition(partition_number, pdf_file, **kwargs):
    ti = kwargs['ti']

    pages_dir = os.path.join(STORAGE_DIR, ENVIRONMENT, '_processed_pages')    
    os.makedirs(pages_dir, exist_ok=True)
    output_file = os.path.join(pages_dir, f'partition_{partition_number}.pkl')

    if os.path.exists(output_file):
        log_progress(ti, f"Skipping partition {partition_number} as it already exists")
        return output_file

    log_progress(ti, f"Processing PDF partition {partition_number}")

    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown"
    )

    reader = PdfReader(pdf_file)
    total_pages = len(reader.pages)
    
    if ENVIRONMENT == 'development':
        total_pages = min(total_pages, DEV_PAGE_LIMIT)
        log_progress(ti, f"Development mode: Processing only the first {DEV_PAGE_LIMIT} pages")
    
    pages_per_partition = max(1, total_pages // NUM_PARTITIONS)
    start_page = partition_number * pages_per_partition
    end_page = min((partition_number + 1) * pages_per_partition, total_pages)

    all_documents = []
    for i in range(start_page, end_page):
        page_docs = process_page(reader.pages[i], i + 1, parser)
        all_documents.extend(page_docs)
        log_progress(ti, f"Processed page {i+1}/{end_page} of partition {partition_number}")

    df = pd.DataFrame({
        'page_number': [doc.metadata['page_number'] for doc in all_documents],
        'content': [doc.text for doc in all_documents]
    })
    df.to_pickle(output_file)

    log_progress(ti, f"Finished processing PDF partition {partition_number}, extracted {len(all_documents)} documents.")

    return output_file

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