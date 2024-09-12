import os
from tqdm import tqdm
import json
from langfuse import Langfuse
from langfuse.decorators import observe
import re
import time
import pandas as pd
from pipeline.llm_client import get_llm#, create_extraction_chain
from pipeline.vectorstore_gene_synonyms import VectorStore_genes

class Extractor:
    langfuse = Langfuse() # add langfuse instance as static element
    def __init__(self, extraction_type, prompt_name, fields, model):
        self.extraction_type = extraction_type
        self.prompt_name = prompt_name
        self.fields = fields
        self.model = model
        
        self.temp_folder = self.create_temp_folder()

    def create_temp_folder(self):
        temp_folder = f'data/temporary_{self.extraction_type}'
        os.makedirs(temp_folder, exist_ok=True)
        return temp_folder

    @observe(as_type="generation")
    def get_llm_response(self, abstract_text):
        prompt = langfuse.get_prompt(self.prompt_name)
        msg = prompt.compile(abstract=abstract_text)

        llm = get_llm(self.model)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                return llm.chat(msg)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 20 * (attempt + 1)
                    print(f"Error occurred: {e}. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Max retries reached. Last error: {e}")
                    raise

    def extract_info(self, response):
        extracted = {}
        for field in self.fields:
            match = re.search(rf'\[{field}\](.*?)\[/{field}\]', response, re.DOTALL)
            extracted[field] = match.group(1).strip() if match else "Error: not found"
        return extracted

    def process_abstracts(self, abstracts_df):
        results = []
        for index, row in tqdm(abstracts_df.iterrows(), total=len(abstracts_df), desc=f"Extracting {self.extraction_type}"):
            abstract_number = row['Abstract Number']
            temp_file = os.path.join(self.temp_folder, f"{abstract_number}.json")
            
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    result = json.load(f)
            else:
                abstract_text = row['Title'] + ' ' + row['Abstract']
                response = self.get_llm_response(abstract_text)
                
                result = {
                    'Abstract Number': abstract_number,
                    'Abstract Text': abstract_text,
                    'response': response,
                    **self.extract_info(response)
                }
                
                with open(temp_file, 'w') as f:
                    json.dump(result, f)
            
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{self.extraction_type}_extraction_results.csv', index=False)
        print(f"{self.extraction_type.capitalize()} extraction completed and results exported to '{self.extraction_type}_extraction_results.csv'")

class IndicationExtractor(Extractor):
    def __init__(self, model='groq'):
        super().__init__('indication', 'GetIndication', ['indication', 'subtype'], model)

class PhaseExtractor(Extractor):
    def __init__(self, model='groq'):
        super().__init__('phase', 'Phase', ['phase', 'model'], model)


class InitialGeneExtractor(Extractor):
    def __init__(self, model='groq'):
        super().__init__('initial_gene', 'all_genes', ['reasoning', 'gene'], model)

    def process_abstracts(self, abstracts_df):
        results = []
        for index, row in tqdm(abstracts_df.iterrows(), total=len(abstracts_df), desc="Extracting initial genes"):
            abstract_number = row['Abstract Number']
            temp_file = os.path.join(self.temp_folder, f"{abstract_number}.json")
            
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    result = json.load(f)
            else:
                abstract_text = row['Title'] + ' ' + row['Abstract']
                response = self.get_llm_response(abstract_text)
                result = create_extraction_chain(response, get_llm(self.model))
                result['Abstract Number'] = abstract_number
                result['Abstract Text'] = abstract_text
                
                with open(temp_file, 'w') as f:
                    json.dump(result, f)
            
            results.append(result)

        return results


class GeneTargetExtractor(Extractor):
    def __init__(self, model='groq'):
        super().__init__('gene_target', 'gene_target', ['reasoning', 'answer'], model)
        self.vectorstore = VectorStore_genes()

    def get_gene_context(self, potential_genes):
        gene_context = []
        symbols_only = []
        for gene in potential_genes:
            context = self.vectorstore.retrieve('gene-index', gene)
            symbols_only.extend([item['symbol'] for item in context])
            gene_context.append(f"{gene}: {context}")
        return gene_context, symbols_only

    def process_abstracts(self, abstracts_df, initial_results):
        results = []
        for index, row in tqdm(abstracts_df.iterrows(), total=len(abstracts_df), desc="Extracting gene targets"):
            abstract_number = row['Abstract Number']
            temp_file = os.path.join(self.temp_folder, f"{abstract_number}.json")
            
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    result = json.load(f)
            else:
                abstract_text = row['Title'] + ' ' + row['Abstract']
                initial_result = initial_results[index]
                potential_genes = initial_result.get('extracted', [])
                
                gene_context, symbols_only = self.get_gene_context(potential_genes)
                
                response = self.get_llm_response(abstract_text, potential_genes, gene_context, symbols_only)
                final_result = create_extraction_chain(response, get_llm(self.model))
                
                result = {
                    'Abstract Number': abstract_number,
                    'Abstract Text': abstract_text,
                    'potential_genes': potential_genes,
                    'initial_reasoning': initial_result.get('reasoning', ''),
                    'gene_context': gene_context,
                    'symbols_only': symbols_only,
                    'final_reasoning': final_result.get('reasoning', ''),
                    'targets': final_result.get('extracted', [])
                }
                
                with open(temp_file, 'w') as f:
                    json.dump(result, f)
            
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{self.extraction_type}_extraction_results.csv', index=False)
        
        print(f"{self.extraction_type.capitalize()} extraction completed and results exported to '{self.extraction_type}_extraction_results.csv'")