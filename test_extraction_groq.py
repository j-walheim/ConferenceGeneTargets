# %% 
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from pipeline.process_abstract import extract_abstract_info
from defs.abstract_class import Author, GeneDisease, Abstract

import os
import getpass
import pickle

from dotenv import load_dotenv
load_dotenv()


# %% 


#
# read pkl /teamspace/studios/this_studio/ConferenceGeneTargets/data/old/processed_airflow/processed_pages/partition_3.pkl
abstract = pickle.load(open('/teamspace/studios/this_studio/ConferenceGeneTargets/data/processed_pages/partition_2.pkl', 'rb'))


tmp = extract_abstract_info(abstract.iloc[0])

