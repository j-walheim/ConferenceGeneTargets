import random
import pandas as pd
from langfuse import Langfuse
from dotenv import load_dotenv
from tqdm import tqdm
from pipeline.llm_client import get_llm
from pipeline.extractor import IndicationExtractor, PhaseExtractor, InitialGeneExtractor, GeneTargetExtractor, ModalityExtractor

# Load environment variables and initialize
load_dotenv()
random.seed(1)
model = 'gpt-4o'
llm = get_llm(model)


for source in ['orals','posters']:
    # Load the CSV file
    abstracts_df = pd.read_csv(f'data/input/abstracts_source.csv')

    abstracts_df = pd.read_csv('data/input/abstracts_talks.csv')

    # Filter abstracts containing 'background' - some extracted rows do not have abstract text
    if(source == 'orals'):
            abstracts_df = abstracts_df[abstracts_df['Abstract'].str.contains('background', case=False, na=False)]



    # Initialize extractors
    modality_extractor = ModalityExtractor(model)
    indication_extractor = IndicationExtractor(model)
    phase_extractor = PhaseExtractor(model)
    initial_gene_extractor = InitialGeneExtractor(model)
    gene_target_extractor = GeneTargetExtractor(model)

    # Initialize results lists
    indication_results = []
    phase_results = []
    initial_gene_results = []
    gene_target_results = []
    modality_results = []

    # Main loop to process abstracts
    for _, row in tqdm(abstracts_df.iterrows(), total=len(abstracts_df), desc="Processing abstracts"):
        abstract_number = row['Abstract Number']
        abstract_text = f"Title: {row['Title']}\n\n{row['Abstract']}"

        # Extract modality
        modality_result = modality_extractor.process_abstract(abstract_number, abstract_text)
        modality_results.append(modality_result)

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

    # Convert results to DataFrames
    modality_df = pd.DataFrame(modality_results)
    indication_df = pd.DataFrame(indication_results)
    phase_df = pd.DataFrame(phase_results)
    initial_gene_df = pd.DataFrame(initial_gene_results)
    gene_target_df = pd.DataFrame(gene_target_results)

    # Merge all results based on 'Abstract Number'
    extraction_results = pd.concat([indication_df, phase_df, initial_gene_df, gene_target_df, modality_df], axis=1)
    extraction_results = extraction_results.loc[:,~extraction_results.columns.duplicated()]

    # Save the merged results to CSV
    extraction_results.to_csv('data/merged_extraction_results.csv', index=False)

    # Print column names of the resulting file
    print("Column names of the merged results:")
    print(extraction_results.columns.tolist())

    print("Extraction completed. Merged results saved to 'data/merged_extraction_results.csv'.")


# # Provide final extraction results for website
# merged_data = pd.read_csv('data/merged_extraction_results.csv')

# # Select and rename the desired columns from merged_data
# selected_columns = {
#     'Abstract Number': 'Abstract Number',
#     'Abstract Text_x': 'Abstract Text',
#     'Extracted Indication': 'Extracted Indication',
#     'Indication Subtype': 'Indication Subtype',
#     'Extracted Phase': 'Extracted Phase',
#     'Preclinical model': 'Preclinical model'
# }

# merged_data = merged_data[list(selected_columns.keys())].rename(columns=selected_columns)

# # Merge extraction_results with merged_data
# final_results = pd.merge(merged_data, extraction_results, on='Abstract Number', how='outer')

# # Save the final results to CSV
# final_results.to_csv('data/final_extraction_results.csv', index=False)

# # Print column names of the resulting file
# print("Column names of the final results:")
# print(final_results.columns.tolist())

# print("Extraction completed. Final results saved to 'data/final_extraction_results.csv'.")