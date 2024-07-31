from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Optional
# TCGA project names to full names mapping
tcga_to_fullname = {
    "TCGA-BRCA": "Invasive Breast Carcinoma",
    "TCGA-GBM": "Glioblastoma",
    "TCGA-OV": "High-Grade Serous Ovarian Cancer",
    "TCGA-LUAD": "Lung Adenocarcinoma",
    "TCGA-UCEC": "Endometrial Carcinoma",
    "TCGA-KIRC": "Renal Clear Cell Carcinoma",
    "TCGA-HNSC": "Head and Neck Squamous Cell Carcinoma",
    "TCGA-LGG": "Diffuse Glioma",
    "TCGA-THCA": "Thyroid Cancer",
    "TCGA-LUSC": "Lung Squamous Cell Carcinoma",
    "TCGA-PRAD": "Prostate Adenocarcinoma",
    "TCGA-SKCM": "Melanoma",
    "TCGA-COAD": "Colorectal Adenocarcinoma",
    "TCGA-STAD": "Stomach Adenocarcinoma",
    "TCGA-BLCA": "Bladder Urothelial Carcinoma",
    "TCGA-LIHC": "Hepatocellular Carcinoma",
    "TCGA-CESC": "Cervical Squamous Cell Carcinoma",
    "TCGA-KIRP": "Papillary Renal Cell Carcinoma",
    "TCGA-SARC": "Soft Tissue Sarcoma",
    "TCGA-LAML": "Acute Myeloid Leukemia",
    "TCGA-PAAD": "Pancreatic Adenocarcinoma",
    "TCGA-ESCA": "Esophageal Cancer",
    "TCGA-PCPG": "Pheochromocytoma and Paraganglioma",
    "TCGA-READ": "Colorectal Adenocarcinoma",
    "TCGA-TGCT": "Testicular Germ Cell Tumor",
    "TCGA-THYM": "Thymoma",
    "TCGA-KICH": "Chromophobe Renal Cell Carcinoma",
    "TCGA-ACC": "Adrenocortical Carcinoma",
    "TCGA-MESO": "Pleural Mesothelioma",
    "TCGA-UVM": "Uveal Melanoma",
    "TCGA-DLBC": "Diffuse Large B-Cell Lymphoma",
    "TCGA-UCS": "Uterine Carcinosarcoma",
    "TCGA-CHOL": "Cholangiocarcinoma"
}

class Author(BaseModel):
    """Information about an author."""
    name: str = Field(description="The full name of the author")
    affiliation: Optional[str] = Field(default=None, description="The institution or organization the author is affiliated with")

class Abstract(BaseModel):
    """Information extracted from an academic abstract."""
    abstract_number: str = Field(description="The unique 4-digit identifier for the abstract, excluding any non-numeric characters")
    title: str = Field(description="The complete title of the abstract, preserving capitalization and any special characters")
    authors: List[Author] = Field(description="A comprehensive list of all authors mentioned, including their full names and institutional affiliations if provided")
    abstract_text: str = Field(description="The entire text of the abstract, including all sections (introduction, methods, results, conclusion), with proper formatting and whitespace preserved")
    disease: Optional[List[str]] = Field(description="A list of specific cancer indications or types mentioned in the abstract, using standardized oncology terminology (e.g., OncoTree nomenclature)")
    gene: Optional[List[str]] = Field(description="A list of genes mentioned in the abstract, using official HUGO gene symbols only. Include only genes directly related to the discussed cancer or research topic")


class GeneDisease(BaseModel):
    """Information about gene-disease interactions."""
    gene: str = Field(description="The name of the gene")
    disease: str = Field(description="The name of the disease")
    description: Optional[str] = Field(default=None, description="The description of the interaction")
    directionality: Optional[str] = Field(default=None, description="The directionality of the interaction")
