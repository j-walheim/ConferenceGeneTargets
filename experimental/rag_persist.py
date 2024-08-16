# %%
import os
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.prompts import ChatPromptTemplate

# 1. Set up gene synonym dictionary
gene_synonyms = {
    "BRCA1": ["breast cancer 1", "BRCC1", "BRCAI", "BRCA1/BRCA2-containing complex"],
    "TP53": ["tumor protein p53", "p53", "LFS1", "TRP53"],
    "EGFR": ["epidermal growth factor receptor", "ERBB", "ERBB1", "HER1"],
    # Add more genes and their synonyms here
}



# 2. Create documents from the gene synonym dictionary
gene_lookup = [
    f"HUGO: {hugo}, Synonyms: {', '.join(synonyms)}"
    for hugo, synonyms in gene_synonyms.items()
]

# 3. Set up vector store
embeddings = MistralAIEmbeddings()
vectorstore_path = "vectorstore_genes"

# Check if the vectorstore already exists
if os.path.exists(vectorstore_path):
    # Load the existing vectorstore
    vectorstore_genes = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
else:
    # Create a new vectorstore and persist it
    vectorstore_genes = Chroma.from_texts(texts=gene_lookup, embedding=embeddings, persist_directory=vectorstore_path)
    vectorstore_genes.persist()

# 4. Set up retriever
retriever_genes = vectorstore_genes.as_retriever(search_type="similarity", search_kwargs={"k": 5})

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
Researchers are also investigating the role of KRAS in various types of cancer.
"""

result = rag_chain.invoke(text)
print(result)

# No need to delete the collection as we're persisting it

# %%
