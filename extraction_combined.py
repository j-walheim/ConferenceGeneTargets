import random
import pandas as pd
from langfuse import Langfuse
from langfuse.decorators import observe
from pipeline.llm_client import get_llm
from dotenv import load_dotenv
import re
import os
import json
import time
import requests
from tqdm import tqdm
from pipeline.extractor import IndicationExtractor, PhaseExtractor

# Load environment variables and initialize
load_dotenv()
random.seed(1)
model = 'gpt-4o-mini'
llm = get_llm(model)
abstracts_df = pd.read_csv('data/abstracts_posters_esmo.csv')
abstracts_df = abstracts_df.head(2)

langfuse = Langfuse()

# Run both extractions
IndicationExtractor(model).process_abstracts(abstracts_df)
PhaseExtractor(model).process_abstracts(abstracts_df)
