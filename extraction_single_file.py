# %% 
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from pipeline.process_abstract import extract_abstract_info
import json
import pandas as pd

from pipeline.process_abstract import process_abstracts_partition
from dotenv import load_dotenv
load_dotenv()


# %% 
# load data/production/processed_pages/page_219.json
import json

# Load the JSON file
with open('/teamspace/studios/this_studio/ConferenceGeneTargets/data/production/processed_pages/page_6324.json') as f:
    abstract = json.load(f)

abstract_text = abstract[0].get('content')


# %%
result = extract_abstract_info(abstract_text,model = 'groq')

abstract_dict = result.dict()
abstract_dict['text'] = abstract_text
abstract_dict['page_number'] = abstract[0].get('page_number')

json_result = json.dumps(abstract_dict, indent=2)



# %%
# save json_result to data/production/processed_pages/page_219.json
with open('/teamspace/studios/this_studio/ConferenceGeneTargets/tmp/page_6324.json', 'w') as f:
    f.write(json_result)
# %%
