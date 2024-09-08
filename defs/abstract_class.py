from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Optional

class AllGenes(BaseModel):
    """Information about all genes."""
    genes: List[str] = Field(description="The list of all genes names in the abstract", max_items=50)

class Author(BaseModel):
    """Information about an author."""
    name: str = Field(description="The full name of the author")
    affiliation: Optional[str] = Field(default=None, description="The institution or organization the author is affiliated with")

class GeneDisease(BaseModel):
    """Information about gene-disease interactions."""
    gene: str = Field(description="The name of the gene")
    disease: str = Field(description="The name of the disease")
    description: Optional[str] = Field(default=None, description="The description of the interaction")
    directionality: Optional[str] = Field(default=None, description="The directionality of the interaction")

# class Abstract(BaseModel):
#     """Information extracted from an academic abstract."""
#     abstract_number: str = Field(description="The unique 4-digit identifier for the abstract, excluding any non-numeric characters")
#     title: str = Field(description="The complete title of the abstract, preserving capitalization and any special characters")
#     authors: Optional[List[Author]] = Field(description="A comprehensive list of all authors mentioned, including their full names and institutional affiliations if provided")
#     disease: Optional[List[str]] = Field(description="A list of specific cancer indications or types mentioned in the abstract")
#     gene: Optional[List[str]] = Field(description="A list of genes mentioned in the abstract.")
# #    interaction: Optional[List[GeneDisease]] = Field(description="A list of gene-disease interactions mentioned in the abstract")
#     gene_target: Optional[str] = Field(description="The primary gene target that impacts the disease and can be modulated to change the disease")
#     organism: Optional[List[str]] = Field(description="The types of organism used in the study. Must be 'cell line', 'PDX', 'animal', 'human', or 'n/a' if not applicable or not specified.")
#     compound_name: Optional[str] = Field(description="The name of the compound or drug mentioned in the abstract. E.g. 'GS-P-328', 'IOMX-0675', or 'Pembrolizumab', or 'n/a' if not applicable or not specified.")

class Target(BaseModel):
    """Information extracted from an academic abstract about the modulated target."""
    Target: Optional[str] = Field(description="The primary gene target that impacts the disease and can be modulated to change the disease")

class Disease(BaseModel):
    """Information extracted from an academic abstract about the disease."""
    Disease: str = Field(description="The primary disease or cancer type discussed in the abstract")