from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.decorators import dag, task
from airflow.models import Variable
from datetime import timedelta
from airflow.utils.task_group import TaskGroup

import os
import requests
from llama_parse import LlamaParse
import pickle
from PyPDF2 import PdfReader, PdfWriter
import logging
import pandas as pd

# Configuration
DEFAULT_ARGS = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': '2023-01-01',
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}
NUM_PARTITIONS = 4
ENVIRONMENT = Variable.get("environment", default_var="development")
#STORAGE_DIR = Variable.get("storage_dir", default_var="/path/to/storage")
STORAGE_DIR = '/teamspace/studios/this_studio/ConferenceGeneTargetsRAG/data/processed_airflow'
MAX_ACTIVE_TASKS = 4
DEV_PAGE_LIMIT = 10

LLAMA_CLOUD_API_KEY = Variable.get("LLAMA_CLOUD_API_KEY")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def log_progress(ti, message):
    ti.xcom_push(key='progress_log', value=message)
    print(message)

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

@dag(
    dag_id=f'pdf_processing_pipeline_{ENVIRONMENT}',
    default_args=DEFAULT_ARGS,
    description=f'A DAG to process PDF documents ({ENVIRONMENT} environment)',
    schedule_interval=None,#timedelta(days=1),
    concurrency=MAX_ACTIVE_TASKS,
)
def ProcessPDFs():
    
    @task
    def download_pdf(url, pdf_path):
        os.makedirs(pdf_path, exist_ok=True)
        fname = os.path.join(pdf_path, os.path.basename(url))
        
        if not os.path.exists(fname):
            response = requests.get(url)
            response.raise_for_status()
            with open(fname, 'wb') as file:
                file.write(response.content)
            logging.info(f"PDF downloaded to {fname}")
        else:
            logging.info("PDF already downloaded.")
        
        return fname

    @task
    def process_pdf_partition(pdf_file, partition_number, **kwargs):
        ti = kwargs['ti']
        pages_dir = os.path.join(STORAGE_DIR, 'processed_pages')
        ensure_dir(pages_dir)
        output_file = os.path.join(pages_dir, f'partition_{partition_number}.pkl')

        if not os.path.exists(output_file):
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
            start_page = (partition_number - 1) * pages_per_partition
            end_page = min(start_page + pages_per_partition, total_pages)

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

            log_progress(ti, f"Finished processing PDF partition {partition_number}")
        else:
            log_progress(ti, f"Skipping partition {partition_number} as it already exists")

        return output_file


#    @task
#    def extractJSON(pdf_file, **kwargs):
        
        
        
        
    @task
    def combine_pdf_partitions(partition_files, **kwargs):
        ti = kwargs['ti']
        log_progress(ti, f"Starting to combine PDF partitions. Retrieved {len(partition_files)} partition files.")
        
        valid_files = [file for file in partition_files if file and os.path.exists(file)]
        log_progress(ti, f"Found {len(valid_files)} valid partition files.")
        
        if not valid_files:
            raise ValueError("No valid partition files found to combine.")
        
        dfs = [pd.read_pickle(file) for file in valid_files]
        combined_df = pd.concat(dfs, ignore_index=True)
        
        output_file = os.path.join(STORAGE_DIR, f'combined_pdf_content_{ENVIRONMENT}.pkl')
        ensure_dir(os.path.dirname(output_file))
        combined_df.to_pickle(output_file)
        
        log_progress(ti, f"Finished combining PDF partitions for {ENVIRONMENT} environment")
        return output_file

    pdf_file = download_pdf(Variable.get("pdf_url",
                                         default_var = 
                                        'https://www.aacr.org/wp-content/uploads/2024/04/AACR2024_Regular_Abstracts_04-01-24.pdf' ), 
                            Variable.get("pdf_path", 
                                         default_var = '/teamspace/studios/this_studio/ConferenceGeneTargetsRAG/data/pdfs'))
    
    with TaskGroup(group_id='process_partitions') as process_group:
        partition_files = [process_pdf_partition(pdf_file, i) for i in range(1, NUM_PARTITIONS + 1)]

    combined_pdf = combine_pdf_partitions(partition_files)

    pdf_file >> process_group >> combined_pdf

dag = ProcessPDFs()
