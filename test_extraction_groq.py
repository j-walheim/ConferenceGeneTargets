# %% 
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from defs.abstract_class import Author, GeneDisease, Abstract

import os
import getpass

# Set up the Groq API key
#os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Initialize the Groq AI model
llm = ChatGroq(
#    model="mixtral-8x7b-32768",
    model='llama3-groq-70b-8192-tool-use-preview',
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# # Define the prompt template
# prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are an expert at extracting information from academic abstracts. 
#     Extract the requested information accurately and completely. 
#     Pay attention to details like author affiliations and ensure the abstract text is complete with proper whitespace."""),
#     ("human", "{text}")
# ])

# Define the improved prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting information from academic abstracts, with a focus on genetics and oncology. Your task is to accurately and completely extract the requested information, paying close attention to details. Extract the information according to the following structure:

    1. abstract_number: Extract ONLY the 4-digit number from the abstract number. Ignore any preceding or following characters. For example, if you see '0009AFNT-212', extract ONLY '0009'. If no 4-digit number is present, use 'n/a'.

    2. title: The full title of the abstract.

    3. authors: A list of authors, each containing:
       - name: The full name of the author
       - affiliation: The institution or organization the author is affiliated with. If not available, use 'n/a'.

    4. abstract_text: The full text of the abstract, with any missing whitespace corrected.

    5. gene_disease: A list of gene-disease interactions, each containing:
       - gene: The name of the gene (use only official HUGO gene symbols)
       - disease: The name of the disease (use OncoTree nomenclature for cancer indications)
       - description: The description of the interaction. If not available, use 'n/a'.
       - directionality: The directionality of the interaction. If not available, use 'n/a'.

    6. gene: A list of genes mentioned in the abstract (use only official HUGO gene symbols).

    Follow these guidelines strictly:
    - For the abstract number, extract ONLY the 4-digit number. Ignore any other characters.
    - For genes, use only official HUGO gene symbols. If a gene is mentioned using a different nomenclature, convert it to the HUGO symbol.
    - For cancer indications, use the OncoTree nomenclature. If the abstract uses a different nomenclature, convert it to the appropriate OncoTree term.
    - If any information is not available or cannot be determined with certainty, use 'n/a' (not applicable) for that field.
    - Never guess or infer information that is not explicitly stated in the abstract. If you are unsure about any information, use 'n/a'.
    - Do not attempt to fill in missing information based on context or general knowledge. If it's not in the abstract, it's 'n/a'.
    - Ensure the abstract text is complete and properly formatted with correct whitespace.

    Accuracy is paramount. It is better to use 'n/a' than to provide potentially incorrect information. Under no circumstances should you guess or make assumptions about any data."""),
    ("human", "{text}")
])

# %% 
# Create the extraction chain
extraction_chain = prompt | llm.with_structured_output(schema=Abstract)

# The abstract text
abstract_text = """'#0009AFNT-212: ATRAC-knocked-in KRASG12D-specific TCR-T cell product enhanced with CD8αβ and a chimeric cytokine receptor for treatment of solid cancers.\n\nA. Drain1, N. Rouillard1, N. Swanson1, M. Canestraro1, S. Narayan1, T. Warner1, N. Danek1, K. Gareau1, J. Liang1, L. Shen1, T. Tetrault1, I. Priyata1, S. Vidyasagar3, S. S. Chandran4, C. A. Klebanoff1, T. Riggins-Walker1, H.-W. Liu1, K. Pechhold1, L. Brown1, J. Francis1, X. He1, P. Browne2, R. Lamothe2, M. Storlie2, G. Cost2, T. M. Schmitt3, P. D. Greenberg\n\n1Affini-T Therapeutics, Inc., Watertown, MA, 24, H. Lam1, A. Gupta1, D. Hallet1, G. Shapiro1, K. Nguyen1,3Fred Hutchinson Cancer Research Center, Seattle, WA,L. Vincent1; 4Memorial Sloan Kettering\n\nCancer Center (MSKCC), New York, NY\n\nThe KRASG12Dmutation is an ideal target for anti-cancer therapies as its expression is typically clonal, restricted to cancer tissue, and is among the most common oncogenic drivers in solid tumors. TCR-T cell therapies have demonstrated clinical activity in some solid cancers but have been limited by heterogeneous antigen expression and unfavorable tumor microenvironments. By targeting the KRASG12Dmutation for which the cancer has established genetic dependency, AFNT-212 is designed to selectively target all cancer cells while avoiding on-target/off-tumor toxicities. AFNT-212 is non-virally engineered to knock-in a 5-transgene cassette expressing a high-avidity TCR specific for the KRASG12Dmutation, a CD8α/β coreceptor, and a chimeric cytokine receptor. Transgene insertion at theTRAClocus disrupts expression of the endogenous TCRα, further enhancing the expression/activity of the transgenic KRASG12D TCR.\n\nPrimary human CD8+and CD4+T cells were genetically engineered by a novel CRISPR-Cas nuclease system to integrate AFNT-212 transgenes within theTRAClocus. A cGMP compatible scale-up process for non-viral knock-in was established to support AFNT-212 clinical manufacturing. The activity of AFNT-212 was assessed against a panel of human KRASG12Dtumor cell linesin vitroand established mouse xenograft modelsin vivo. The preclinical safety profile of AFNT-212 was evaluated by X-scan and crossreactivity assessment, alloreactivity studies, and cytokine independent growth studies. The specificity of gene-editing (GE) was assessed by an unbiased oligo-capture method followed by targeted sequencing.\n\nAFNT-212 TCR-T cells demonstrated potentin vitroanti-tumor activity against endogenously expressing HLA-A*11:01 KRASG12Dtumor cells, including during chronic exposure to viable tumor cells. AFNT-212 TCR-T cells showed robust antitumor activity in established xenograft mouse modelsin vivo. No cross-reactivity was identified for the KRASG12DTCR against potential self-peptides even at supraphysiological levels, demonstrating high specificity of the TCR. No alloreactivity or cytokine-independent proliferation was observed. GE safety evaluations did not reveal any off-target activity using high sensitivity (~0.1%) NGS-based analyses or any GE-associated chromosomal rearrangements. The manufacturing of AFNT-212 consistently delivered >50-fold expansion of engineered TCR-T cells to meet expected clinical dose levels and exhibit memory/stemness phenotypes and negligible markers of immunologic exhaustion.\n\nAFNT-212, a novel TCR T cell therapy targeting KRASG12Dmutant tumors, demonstrates robust activity against KRASG12Dmutant tumorsin vitroandin vivo. The robust manufacturing process developed using non-viral gene editing in theTRAClocus will support future clinical development of AFNT-212.'"""

# Extract information
result = extraction_chain.invoke({"text": abstract_text})

print(result)
# %%
