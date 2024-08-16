import os
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from utils.get_disease_terms import prepare_disease_synonyms
from utils.get_gene_synonyms import prepare_gene_synonyms
from langchain_community.embeddings import HuggingFaceEmbeddings



# Prepare disease and gene synonyms
disease_synonyms = prepare_disease_synonyms()
gene_synonyms = prepare_gene_synonyms()

gene_lookup = [
    f"HUGO: {row['Symbol']}{', Synonyms: ' + row['Synonyms'] if row['Synonyms'] != '-' else ''}"    for _, row in gene_synonyms.iterrows()
]


# Create documents from the disease synonym dictionary
disease_lookup = [
    f"Disease: {disease}, Synonyms: {', '.join(synonyms)}"
    for disease, synonyms in disease_synonyms.items()
]

# %%

# Set up vector stores
#embeddings = MistralAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Gene vectorstore
gene_vectorstore_path = "vectorstore_genes"
if os.path.exists(gene_vectorstore_path):
    print(f"Loading existing gene vectorstore from {gene_vectorstore_path}")
    vectorstore_genes = Chroma(persist_directory=gene_vectorstore_path, embedding_function=embeddings)
else:
    print(f"Creating new gene vectorstore at {gene_vectorstore_path}")
    vectorstore_genes = Chroma.from_texts(texts=gene_lookup, embedding=embeddings, persist_directory=gene_vectorstore_path)
    print("Persisting gene vectorstore")
    vectorstore_genes.persist()
    print("Gene vectorstore persisted successfully")


# Disease vectorstore
disease_vectorstore_path = "vectorstore_diseases"
if os.path.exists(disease_vectorstore_path):
    print(f"Loading existing disease vectorstore from {disease_vectorstore_path}")
    vectorstore_diseases = Chroma(persist_directory=disease_vectorstore_path, embedding_function=embeddings)
else:
    print(f"Creating new disease vectorstore at {disease_vectorstore_path}")
    vectorstore_diseases = Chroma.from_texts(texts=disease_lookup, embedding=embeddings, persist_directory=disease_vectorstore_path)
    print("Persisting disease vectorstore")
    vectorstore_diseases.persist()
    print("Disease vectorstore persisted successfully")
# Set up retrievers
retriever_genes = vectorstore_genes.as_retriever(search_type="similarity", search_kwargs={"k": 5})
retriever_diseases = vectorstore_diseases.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up LLM
llm = ChatMistralAI(model='open-mistral-nemo')

# Set up prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a gene and disease extraction and standardization assistant.
    Your task is to identify genes and diseases mentioned in the given text and map them to their standardized terms.
    Use the following retrieved context to help identify genes, diseases, and their standard names:

    Gene context:
    {context_genes}

    Disease context:
    {context_diseases}

    If you identify a gene or disease mentioned in the text, provide its standardized term.
    If you're unsure about a gene or disease mention, don't include it in the results.
    """),
    ("human", "Extract genes and diseases from the following text. Use gene and disease symbols as indicated above :\n\n{input}"),
])

# Create the RAG chain
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context_genes": retriever_genes | format_docs,
        "context_diseases": retriever_diseases | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Example usage
text = """
 "# TUMOR BIOLOGY: Stem Cells in Tumor Initiation and Progression\n\n# Poster Session\n\n|#0248|HSPA9/mortalin regulates erythroid maturation through a TP53-dependent mechanism in human CD34+ hematopoietic progenitor cells.| |\n|---|---|---|\n|T. Liu1, C. Butler1, M. Dunmire1, G. Szalai2, A. Johnson2, J. Choi3, M. Walter3;| | |\n|1West Virginia School of Osteopathic Medicine, Lewisburg, WV,|2Burrell College of Osteopathic Medicine, Las Cruces, NM,|3Washington University in St. Louis, St. Louis, MO|\n\nMortalin, encoded by the HSPA9 gene, is a highly conserved heat-shock chaperone belonging to the HSP70 family. It is predominantly presented in the mitochondria, and is critical in regulating a variety of cell physiological functions such as response to cell stress, control of cell proliferation, and inhibition/prevention of apoptosis. Myelodysplastic syndromes (MDS) are a group of hematopoietic stem cell malignancies characterized by ineffective hematopoiesis, increased apoptosis of bone marrow cells, and anemia. Up to 25% of MDS patients harbor an interstitial deletion on the long arm of chromosome 5, also known as del(5q), creating haploinsufficiency for multiple genes including HSPA9. Our prior study showed that knockdown of HSPA9 induces TP53-dependent apoptosis in human CD34+ hematopoietic progenitor cells, consistent with cytopenia observed in MDS patients. Since anemia is another featured symptom of MDS, we hypothesize that HSPA9 plays a role in regulating erythroid maturation. To test our hypothesis, we inhibited the expression of HSPA9 using various methods and measured the erythroid maturation in human CD34+ cells. We used siRNA targeting HSPA9 and found that HSPA9 siRNA significantly inhibited the cell growth, increased cell apoptosis, inhibited erythroid maturation (using CD71 as a surrogate marker), and increased p53 expression (p<0.01) compared to control siRNA in human CD34+ cells. Pharmacologic inhibition of HSPA9 by the chemical MKT-077, an inhibitor of HSP70 protein family members including mortalin, also increased p53 expression and inhibited erythroid maturation in human CD34+ cells. In addition, knockdown of HSPA9 by shRNA showed significant inhibition of erythroid maturation in human CD34+ cells compared to control shRNA. In order to test whether the regulation of erythroid maturation by HSPA9 is TP53-dependent or not, we constructed shRNAs targeting TP53 genes and simultaneously transduced lentivirus containing shRNAs targeting HSPA9 and TP53 respectively in human CD34+ cells using double antibiotics selection (puromycin for shRNA targeting HSPA9 and hygromycin for shRNA targeting TP53). We found that TP53 knockdown partially rescued the erythroid maturation defect caused by HSPA9 inhibition, suggesting that erythroid maturation inhibition by HSPA9 knockdown is partly mediated through a TP53 mechanism. Collectively, our results suggest that the increased apoptosis and reduced erythroid maturation observed in del(5a)-associated MDS is TP53-dependent. HSPA9/mortalin may be a potential target to treat anemia in del(5q) MDS patients, although simultaneous loss of multiple genes on del(5q) likely contributes to the complex phenotypes observed in MDS. Thus, our study not only uncovers some underlying mechanisms of del(5q) MDS, but also provides potential therapeutic indications through gene targeting in clinical MDS treatment."
"""

result = rag_chain.invoke(text)
print(result)

# %%