import os
import dotenv
import sys
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from datetime import timedelta
from airflow.utils.task_group import TaskGroup
import sys
sys.path.append('/teamspace/studios/this_studio/ConferenceGeneTargets')

from RAG.vectorstore_ontology import VectorStore
from pipeline.ingestion_pdf import process_pdf_partition, combine_pdf_partitions, download_pdf_task, partition_into_batches
from pipeline.config import NUM_PARTITIONS, MAX_ACTIVE_TASKS, STORAGE_DIR, ENVIRONMENT
dotenv.load_dotenv()
from pipeline.process_abstract import process_abstracts_partition, process_and_merge_abstracts

import json
from PyPDF2 import PdfReader


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





with DAG(
    dag_id=f'pdf_processing_pipeline_{ENVIRONMENT}',
    default_args=DEFAULT_ARGS,
    description=f'A DAG to process PDF documents ({ENVIRONMENT} environment)',
    schedule_interval=None,
    concurrency=MAX_ACTIVE_TASKS,
) as dag:

    
    pdf_file = download_pdf_task()
    
    batches = partition_into_batches(pdf_file, batch_size=100)

    with TaskGroup(group_id='process_partitions') as process_group:
        partition_tasks = process_pdf_partition.expand(batch_pages=batches)
                            
    with TaskGroup(group_id='parse_partitions') as abstract_group:
        partition_tasks = process_abstracts_partition.expand(batch_pages=batches)
        
    #with TaskGroup(group_id='process_abstracts') as abstract_group:
    #     abstract_tasks = [process_abstracts_partition.override(task_id=f'process_abstracts_{i}')(partition_file=partition_tasks[i]) for i in range(NUM_PARTITIONS)]

    # merged_abstracts = process_and_merge_abstracts(abstract_tasks)

#    combined_pdf = combine_pdf_partitions()

    pdf_file >> process_group >> abstract_group# >> merged_abstracts #>> combined_pdf
#    pdf_file >> abstract_group 