from airflow import DAG

from airflow.decorators import task
from airflow.models import Variable
from datetime import timedelta
from airflow.utils.task_group import TaskGroup
import os
import dotenv
import sys
sys.path.append('/teamspace/studios/this_studio/ConferenceGeneTargets')

from pipeline.ingestion_pdf import process_pdf_partition, combine_pdf_partitions, download_pdf_task
from pipeline.config import NUM_PARTITIONS, MAX_ACTIVE_TASKS
dotenv.load_dotenv()

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


ENVIRONMENT = os.getenv("environment", "development")
STORAGE_DIR = os.getenv("storage_dir", "/teamspace/studios/this_studio/ConferenceGeneTargetsRAG/data/processed_airflow")


with DAG(
    dag_id=f'pdf_processing_pipeline_{ENVIRONMENT}',
    default_args=DEFAULT_ARGS,
    description=f'A DAG to process PDF documents ({ENVIRONMENT} environment)',
    schedule_interval=None,
    concurrency=MAX_ACTIVE_TASKS,
) as dag:

    
    pdf_file = download_pdf_task()


    with TaskGroup(group_id='process_partitions') as process_group:
        partition_tasks = [process_pdf_partition.override(task_id=f'process_partition_{i}')(partition_number=i, pdf_file=pdf_file) for i in range(0, NUM_PARTITIONS)]


    combined_pdf = combine_pdf_partitions()

    pdf_file >> process_group >> combined_pdf