# %%
import os
from langfuse import Langfuse
from pipeline.llm_client import get_llm
from pipeline.extractor import PhaseExtractor
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


# Load environment variables
load_dotenv()

# Initialize Langfuse
langfuse = Langfuse()

# Initialize LLM
model = 'gpt-4o-mini'
llm = get_llm(model)


try:
    dataset = langfuse.get_dataset("clinical_trial_phases")
except Exception as e:
    if "Dataset not found" in str(e):
        # Create a dataset
        langfuse.create_dataset(name="clinical_trial_phases")
        dataset = langfuse.get_dataset("clinical_trial_phases")
        
        examples = pd.read_csv('/Users/jonas/Documents/python/ConferenceGeneTargets/data/phase_df.csv')
        examples = examples.head(15)

        # Add items to the dataset
        for index, row in examples.iterrows():
            langfuse.create_dataset_item(
                dataset_name="clinical_trial_phases",
                input={"abstract": row['Abstract Text']},
                expected_output=row['Extracted Phase']  
            )    
    else:
        raise

# %%


# Define evaluation function
def evaluate_phase(output, expected_output):
    return output.lower() == expected_output.lower()

# Run experiment
def run_experiment(experiment_name, prompt_name):
    dataset = langfuse.get_dataset("clinical_trial_phases")
    extractor = PhaseExtractor(model)

    for item in tqdm(dataset.items):
        abstract_text = item.input["abstract"]
        
        # Use the PhaseExtractor to get the phase
        result = extractor.process_abstract("test", abstract_text,overwriteExisting = True)
        extracted_phase = result.get('phase', '')

        # Create a generation object
        langfuse_generation = langfuse.generation(
            name="extract-clinical-phase",
            input=abstract_text,
            output=result,
            model=model
        )

        # Link the item to the generation
        item.link(langfuse_generation, experiment_name)

        # Score the generation
        langfuse_generation.score(
            name="exact_match",
            value=evaluate_phase(extracted_phase, item.expected_output)
        )

# %%

# Run experiments with different prompts
run_experiment("gpt4o-mini", "Phase")
#run_experiment("improved_prompt", "ImprovedPhase")

print("Experiments completed. Check the Langfuse UI for results.")
