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
    abstract_number: str = Field(description="The 4-digit abstract number")
    title: str = Field(description="The full title of the abstract")
    authors: List[Author] = Field(description="List of authors and their affiliations")
    abstract_text: str = Field(description="The full text of the abstract, with any missing whitespace corrected")
    gene_disease: Optional[List[GeneDisease]] = Field(description="List of genes mentioned in the abstract")
    gene: Optional[List[str]] = Field(description="List of genes mentioned in the abstract")
    
    
