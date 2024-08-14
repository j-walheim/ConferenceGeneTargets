import os
MAX_ACTIVE_TASKS = 4
NUM_PARTITIONS = 20
DEV_PAGE_LIMIT = 10
ENVIRONMENT='production'
STORAGE_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')