# %%

from langchain_experimental.llms.ollama_functions import OllamaFunctions


from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Optional

# %%
llm = OllamaFunctions(model="phi3", format = 'json')
#llm = OllamaFunctions(model="mistral",format="json")

# %%

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# ai_msg


# %%
from langchain_core.pydantic_v1 import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])

# %%
ai_msg = llm_with_tools.invoke(
    "what is the weather like in San Francisco",
)
ai_msg

# %%
# Define the schema

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting information from academic abstracts. 
    Extract the requested information accurately and completely. 
    Pay attention to details like author affiliations and ensure the abstract text is complete with proper whitespace."""),
    ("human", "{text}")
])


# Create the extraction chain
from defs.abstract_class import Author, GeneDisease, Abstract

extraction_chain = prompt | llm.bind_tools([Abstract])

# The abstract text
abstract_text = """#0009 AFNT-212: A TRAC-knocked-in KRASG12D-specific TCR-T cell product enhanced with CD8αβ and a chimeric cytokine receptor for treatment of solid cancers.

A. Drain1, N. Rouillard1, N. Swanson1, M. Canestraro1, S. Narayan1, T. Warner1, N. Danek1, K. Gareau1, J. Liang1, L. Shen1, T. Tetrault1, I. Priyata1, S. Vidyasagar1, T. Riggins-Walker1, H.-W. Liu1, K. Pechhold1, L. Brown1, J. Francis1, X. He1, P. Browne2, R. Lamothe2, M. Storlie2, G. Cost2, T. M. Schmitt3, P. D. Greenberg3, S. S. Chandran4, C. A. Klebanoff4, H. Lam1, A. Gupta1, D. Hallet1, G. Shapiro1, K. Nguyen1, L. Vincent1;

1Affini-T Therapeutics, Inc., Watertown, MA, 2Metagenomi, Emeryville, CA, 3Fred Hutchinson Cancer Research Center, Seattle, WA, 4Memorial Sloan Kettering Cancer Center (MSKCC), New York, NY

The KRASG12D mutation is an ideal target for anti-cancer therapies as its expression is typically clonal, restricted to cancer tissue, and is among the most common oncogenic drivers in solid tumors. TCR-T cell therapies have demonstrated clinical activity in some solid cancers but have been limited by heterogeneous antigen expression and unfavorable tumor microenvironments. By targeting the KRASG12D mutation for which the cancer has established genetic dependency, AFNT-212 is designed to selectively target all cancer cells while avoiding on-target/off-tumor toxicities. AFNT-212 is non-virally engineered to knock-in a 5-transgene cassette expressing a high-avidity TCR specific for the KRASG12D mutation, a CD8α/β coreceptor, and a chimeric cytokine receptor. Transgene insertion at the TRAC locus disrupts expression of the endogenous TCRα, further enhancing the expression/activity of the transgenic KRASG12D TCR."""

# Extract information
try:
    result = extraction_chain.invoke({"text": abstract_text})
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")