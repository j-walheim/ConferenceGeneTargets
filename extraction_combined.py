import random
import pandas as pd
from langfuse import Langfuse
from dotenv import load_dotenv
from tqdm import tqdm
from pipeline.llm_client import get_llm
from pipeline.extractor import IndicationExtractor, PhaseExtractor, InitialGeneExtractor, GeneTargetExtractor

# Load environment variables and initialize
load_dotenv()
random.seed(1)
model = 'gpt-4o-mini'
llm = get_llm(model)

# Load the CSV file
abstracts_df = pd.read_csv('data/abstracts_posters_esmo.csv')
abstracts_df = abstracts_df.head(2)  # For testing, remove this line for full dataset

# Initialize extractors
indication_extractor = IndicationExtractor(model)
phase_extractor = PhaseExtractor(model)
initial_gene_extractor = InitialGeneExtractor(model)
gene_target_extractor = GeneTargetExtractor(model)

# Initialize results lists
indication_results = []
phase_results = []
initial_gene_results = []
gene_target_results = []

# Main loop to process abstracts
for _, row in tqdm(abstracts_df.iterrows(), total=len(abstracts_df), desc="Processing abstracts"):
    abstract_number = row['Abstract Number']
    abstract_text = f"Title: {row['Title']}\n\n{row['Abstract']}"

    # # Extract indication
    indication_result = indication_extractor.process_abstract(abstract_number, abstract_text)
    indication_results.append(indication_result)

    # Extract phase
    phase_result = phase_extractor.process_abstract(abstract_number, abstract_text)
    phase_results.append(phase_result)

    # Extract initial genes
    initial_gene_result = initial_gene_extractor.process_abstract(abstract_number, abstract_text)
    initial_gene_results.append(initial_gene_result)

    # Extract gene targets
    gene_target_result = gene_target_extractor.process_abstract(abstract_number, abstract_text, initial_gene_result)
    gene_target_results.append(gene_target_result)

# Convert results to DataFrames and save to CSV
pd.DataFrame(indication_results).to_csv('data/indication_extraction_results.csv', index=False)
pd.DataFrame(phase_results).to_csv('data/phase_extraction_results.csv', index=False)
pd.DataFrame(initial_gene_results).to_csv('data/initial_gene_extraction_results.csv', index=False)
pd.DataFrame(gene_target_results).to_csv('data/gene_target_extraction_results.csv', index=False)

print("Extraction completed. Results saved to CSV files.")
