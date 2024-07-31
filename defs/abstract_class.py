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
    abstract_text: str = Field(description="The entire text of the abstract, including all sections (introduction, methods, results, conclusion), with proper formatting and whitespace preserved")
    disease: Optional[List[str]] = Field(description="A list of specific cancer indications or types mentioned in the abstract, using standardized oncology terminology (e.g., OncoTree nomenclature)")
    gene: Optional[List[str]] = Field(description="A list of genes mentioned in the abstract, using official HUGO gene symbols only. Include only genes directly related to the discussed cancer or research topic")
