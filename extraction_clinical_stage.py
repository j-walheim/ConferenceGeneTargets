# %%
import random
import pandas as pd
from langfuse import Langfuse
from pipeline.utils import get_llm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set random seed
random.seed(1)
model = 'gpt-4o'
llm = get_llm(model)

# Load the CSV file
abstracts_df = pd.read_csv('data/abstracts_posters_esmo.csv')

# Get random 20 rows
abstracts_df = abstracts_df.sample(n=20)

# Select a specific abstract
abstract_text = abstracts_df.iloc[2]['Title'] + abstracts_df.iloc[2]['Abstract']

# Initialize Langfuse
langfuse = Langfuse()


# %%
@observe(as_type="generation")
def get_phase(abstract_text):
    prompt = langfuse.get_prompt("Phase")
    msg = prompt.compile(abstract= abstract_text)

    # Get the response from the LLM
    response = llm.chat(msg)
    return(response)

print(get_phase(abstract_text))
# %%
