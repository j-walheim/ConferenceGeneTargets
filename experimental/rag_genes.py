# %%

import os
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings

from langchain.prompts import ChatPromptTemplate

# 1. Set up gene synonym dictionary
gene_synonyms = {
    "BRCA1": ["breast cancer 1", "BRCC1", "BRCAI", "BRCA1/BRCA2-containing complex"],
    "TP53": ["tumor protein p53", "p53", "LFS1", "TRP53"],
    "EGFR": ["epidermal growth factor receptor", "ERBB", "ERBB1", "HER1"],
    # Add more genes and their synonyms here
}

cancer_lookup = {
    "ADRENAL_GLAND": ["Adrenal Gland Cancer", "Adrenocortical Carcinoma", "Adrenal Cortical Carcinoma", "ACC"],
    "BILLIARY_TRACT": ["Biliary Tract Cancer", "Cholangiocarcinoma", "Bile Duct Cancer", "Gallbladder Cancer"],
    "BONE": ["Bone Cancer", "Osteosarcoma", "Ewing Sarcoma", "Chondrosarcoma", "Bone Sarcoma"],
    "BREAST": ["Breast Cancer", "Mammary Carcinoma", "Ductal Carcinoma", "Lobular Carcinoma", "DCIS", "LCIS"],
    "CNS/BRAIN": ["Central Nervous System Cancer", "Brain Cancer", "Glioma", "Meningioma", "Astrocytoma", "Glioblastoma", "GBM", "Medulloblastoma"],
    "COLORECTAL": ["Colorectal Cancer", "Colon Cancer", "Rectal Cancer", "Bowel Cancer", "CRC"],
    "ESOPHAGUS/STOMACH": ["Esophageal Cancer", "Stomach Cancer", "Gastric Cancer", "Gastroesophageal Junction Cancer", "GEJ Cancer"],
    "HEAD_AND_NECK": ["Head and Neck Cancer", "Oral Cancer", "Laryngeal Cancer", "Pharyngeal Cancer", "Nasopharyngeal Cancer", "Oropharyngeal Cancer"],
    "KIDNEY": ["Kidney Cancer", "Renal Cell Carcinoma", "RCC", "Renal Cancer", "Wilms Tumor", "Nephroblastoma"],
    "LIVER": ["Liver Cancer", "Hepatocellular Carcinoma", "HCC", "Hepatoma", "Primary Liver Cancer"],
    "NSCLC": ["Non-Small Cell Lung Cancer", "NSCLC", "Lung Adenocarcinoma", "Squamous Cell Lung Cancer", "Large Cell Lung Cancer"],
    "SCLC": ["Small Cell Lung Cancer", "SCLC", "Oat Cell Lung Cancer"],
    "LYMPHOID": ["Lymphoma", "Non-Hodgkin Lymphoma", "Hodgkin Lymphoma", "NHL", "HL", "B-cell Lymphoma", "T-cell Lymphoma", "DLBCL"],
    "MYELOID": ["Leukemia", "Acute Myeloid Leukemia", "AML", "Chronic Myeloid Leukemia", "CML", "Myelodysplastic Syndrome", "MDS", "Myeloproliferative Neoplasm", "MPN"],
    "OVARY/FALLOPIAN_TUBE": ["Ovarian Cancer", "Fallopian Tube Cancer", "Primary Peritoneal Cancer", "Epithelial Ovarian Cancer", "Ovarian Germ Cell Tumor"],
    "PANCREAS": ["Pancreatic Cancer", "Pancreatic Adenocarcinoma", "PDAC", "Pancreatic Neuroendocrine Tumor", "PNET"],
    "PROSTATE": ["Prostate Cancer", "Prostatic Adenocarcinoma", "Prostate Adenocarcinoma", "PCa"],
    "SKIN": ["Skin Cancer", "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", "BCC", "SCC", "Merkel Cell Carcinoma"],
    "SOFT_TISSUE": ["Soft Tissue Sarcoma", "Liposarcoma", "Leiomyosarcoma", "Rhabdomyosarcoma", "Synovial Sarcoma", "GIST"],
    "TESTIS": ["Testicular Cancer", "Germ Cell Tumor", "Seminoma", "Non-seminoma", "Testicular Germ Cell Tumor", "TGCT"],
    "THYMUS": ["Thymus Cancer", "Thymoma", "Thymic Carcinoma", "Thymic Neoplasm"],
    "THYROID": ["Thyroid Cancer", "Papillary Thyroid Cancer", "Follicular Thyroid Cancer", "Medullary Thyroid Cancer", "Anaplastic Thyroid Cancer", "PTC", "FTC", "MTC", "ATC"],
    "UTERUS": ["Uterine Cancer", "Endometrial Cancer", "Uterine Sarcoma", "Cervical Cancer", "Uterine Corpus Cancer"]
}




# 2. Create documents from the gene synonym dictionary
gene_lookup = [
    f"HUGO: {hugo}, Synonyms: {', '.join(synonyms)}"
    for hugo, synonyms in gene_synonyms.items()
]

# 3. Set up vector store
embeddings = MistralAIEmbeddings()
vectorstore_genes = Chroma.from_texts(texts=gene_lookup, embedding=embeddings)

# 4. Set up retriever
retriever_genes = vectorstore_genes.as_retriever(search_type="similarity", search_kwargs={"k": 5})
# %%

# 5. Set up LLM
llm = ChatMistralAI(model='open-mistral-nemo')

# 6. Set up prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a gene extraction and standardization assistant. 
    Your task is to identify genes mentioned in the given text and map them to their standardized HUGO terms.
    Use the following retrieved context to help identify genes and their standard names:

    {context_genes}
    
    {context_disease}
    

    If you identify a gene mentioned in the text, provide its standardized HUGO term.
    If you're unsure about a gene mention, don't include it in the results.
    """),
    ("human", "Extract genes from the following text and provide their HUGO terms:\n\n{input}"),
])

# %%

# 7. Create the RAG chain
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context_genes": retriever_genes | format_docs, "input": RunnablePassthrough()} |
    {"context_disease": retriever_genes | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Example usage
text = """
Recent studies have shown that mutations in the BRCA1 gene are associated with an increased risk of breast cancer. 
Additionally, the tumor suppressor gene p53 plays a crucial role in preventing cancer development. 
Researchers are also investigating the role of ERBB1 in various types of cancer.
"""

result = rag_chain.invoke(text)
print(result)

# Cleanup (optional)
vectorstore_genes.delete_collection()
# %%
