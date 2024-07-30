#%%
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from defs.abstract_class import Author, GeneDisease, Abstract

#client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"))
model = 'open-mistral-7b'
model = 'mistral-small-latest'
with open('../.keys/mistral', 'r') as file:
    os.environ['MISTRAL_API_KEY'] = file.read().strip()
# %% 
# Initialize the Mistral AI model
llm = ChatMistralAI(model=model, temperature=0)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting information from academic abstracts. 
    Extract the requested information accurately and completely. 
    Pay attention to details like author affiliations and ensure the abstract text is complete with proper whitespace."""),
    ("human", "{text}")
])

# Create the extraction chain
extraction_chain = prompt | llm.with_structured_output(schema=Abstract)

# The abstract text
abstract_text = """'#0009AFNT-212: ATRAC-knocked-in KRASG12D-specific TCR-T cell product enhanced with CD8αβ and a chimeric cytokine receptor for treatment of solid cancers.\n\nA. Drain1, N. Rouillard1, N. Swanson1, M. Canestraro1, S. Narayan1, T. Warner1, N. Danek1, K. Gareau1, J. Liang1, L. Shen1, T. Tetrault1, I. Priyata1, S. Vidyasagar3, S. S. Chandran4, C. A. Klebanoff1, T. Riggins-Walker1, H.-W. Liu1, K. Pechhold1, L. Brown1, J. Francis1, X. He1, P. Browne2, R. Lamothe2, M. Storlie2, G. Cost2, T. M. Schmitt3, P. D. Greenberg\n\n1Affini-T Therapeutics, Inc., Watertown, MA, 24, H. Lam1, A. Gupta1, D. Hallet1, G. Shapiro1, K. Nguyen1,3Fred Hutchinson Cancer Research Center, Seattle, WA,L. Vincent1; 4Memorial Sloan Kettering\n\nMetagenomi, Emeryville, CA,\n\nCancer Center (MSKCC), New York, NY\n\nThe KRASG12Dmutation is an ideal target for anti-cancer therapies as its expression is typically clonal, restricted to cancer tissue, and is among the most common oncogenic drivers in solid tumors. TCR-T cell therapies have demonstrated clinical activity in some solid cancers but have been limited by heterogeneous antigen expression and unfavorable tumor microenvironments. By targeting the KRASG12Dmutation for which the cancer has established genetic dependency, AFNT-212 is designed to selectively target all cancer cells while avoiding on-target/off-tumor toxicities. AFNT-212 is non-virally engineered to knock-in a 5-transgene cassette expressing a high-avidity TCR specific for the KRASG12Dmutation, a CD8α/β coreceptor, and a chimeric cytokine receptor. Transgene insertion at theTRAClocus disrupts expression of the endogenous TCRα, further enhancing the expression/activity of the transgenic KRASG12D TCR.\n\nPrimary human CD8+and CD4+T cells were genetically engineered by a novel CRISPR-Cas nuclease system to integrate AFNT-212 transgenes within theTRAClocus. A cGMP compatible scale-up process for non-viral knock-in was established to support AFNT-212 clinical manufacturing. The activity of AFNT-212 was assessed against a panel of human KRASG12Dtumor cell linesin vitroand established mouse xenograft modelsin vivo. The preclinical safety profile of AFNT-212 was evaluated by X-scan and crossreactivity assessment, alloreactivity studies, and cytokine independent growth studies. The specificity of gene-editing (GE) was assessed by an unbiased oligo-capture method followed by targeted sequencing.\n\nAFNT-212 TCR-T cells demonstrated potentin vitroanti-tumor activity against endogenously expressing HLA-A*11:01 KRASG12Dtumor cells, including during chronic exposure to viable tumor cells. AFNT-212 TCR-T cells showed robust antitumor activity in established xenograft mouse modelsin vivo. No cross-reactivity was identified for the KRASG12DTCR against potential self-peptides even at supraphysiological levels, demonstrating high specificity of the TCR. No alloreactivity or cytokine-independent proliferation was observed. GE safety evaluations did not reveal any off-target activity using high sensitivity (~0.1%) NGS-based analyses or any GE-associated chromosomal rearrangements. The manufacturing of AFNT-212 consistently delivered >50-fold expansion of engineered TCR-T cells to meet expected clinical dose levels and exhibit memory/stemness phenotypes and negligible markers of immunologic exhaustion.\n\nAFNT-212, a novel TCR T cell therapy targeting KRASG12Dmutant tumors, demonstrates robust activity against KRASG12Dmutant tumorsin vitroandin vivo. The robust manufacturing process developed using non-viral gene editing in theTRAClocus will support future clinical development of AFNT-212.'"""
#abstract_text = """asdasd """

# Extract information
result = extraction_chain.invoke({"text": abstract_text})

#%%
print(result)




# %%
