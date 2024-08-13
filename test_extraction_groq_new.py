# %% 
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from pipeline.process_abstract import extract_abstract_info_mistral
from defs.abstract_class import Author, GeneDisease, Abstract

import os
import getpass
import pickle
import pandas as pd

from pipeline.process_abstract import process_abstracts_partition
from dotenv import load_dotenv
load_dotenv()


# %% 
abstract = pickle.load(open('/teamspace/studios/this_studio/ConferenceGeneTargets/data_old/old/all_abstracts.pkl','rb'))

# %%
df = pd.DataFrame({
    'page_number': [doc.metadata['page_number'] for doc in abstract],
    'content': [doc.text for doc in abstract]
})

# %%
tmp = extract_abstract_info_mistral(df.iloc[0])

#tmp = extract_abstract_info(abstract.iloc[0])

# %%
tmp = process_abstracts_partition