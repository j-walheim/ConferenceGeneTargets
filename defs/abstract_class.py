from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Optional

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

class Abstract(BaseModel):
    """Information extracted from an academic abstract."""
    abstract_number: str = Field(description="The unique 4-digit identifier for the abstract, excluding any non-numeric characters")
    title: str = Field(description="The complete title of the abstract, preserving capitalization and any special characters")
    authors: List[Author] = Field(description="A comprehensive list of all authors mentioned, including their full names and institutional affiliations if provided")
    disease: Optional[List[str]] = Field(description="A list of specific cancer indications or types mentioned in the abstract")
    gene: Optional[List[str]] = Field(description="A list of genes mentioned in the abstract.")
    interaction: Optional[List[GeneDisease]] = Field(description="A list of gene-disease interactions mentioned in the abstract")
    organism: Optional[List[str]] = Field(description="The types of organism used in the study. Must be 'cell line', 'PDX', 'animal', 'human', or 'n/a' if not specified.")
    trial_stage: Optional[List[str]] = Field(description="The stage of the trial or study. Must be 'preclinical', 'Phase I', 'Phase II', 'Phase III', 'post-approval'")
    compound_name: Optional[str] = Field(description="The name of the compound or drug mentioned in the abstract. E.g. 'GS-P-328', 'IOMX-0675', or 'Pembrolizumab'")
    abstract_category: str = Field(description="Assign the abstract to one of the following categories: 'Diagnostics', 'Gene-Disease Associations', or 'Other'. Choose the most appropriate category based on the main focus of the abstract.")

