import os
import json
from langfuse import Langfuse
from langfuse.decorators import observe
import re
import time
from pipeline.llm_client import get_llm
from pipeline.vectorstore_gene_synonyms import VectorStore_genes

class Extractor:
    langfuse = Langfuse()
    
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
    def get_llm_response(self, msg):
        llm = get_llm(self.model)
        return llm.chat(msg)


    def extract_info(self, response):
        extracted = {}
        for field in self.fields:
            match = re.search(rf'\[{field}\](.*?)\[/{field}\]', response, re.DOTALL)
            extracted[field] = match.group(1).strip() if match else "n/a"
        return extracted

    def process_abstract(self, abstract_number, abstract_text):
        temp_file = os.path.join(self.temp_folder, f"{abstract_number}.json")
        
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                return json.load(f)
        
        prompt = self.langfuse.get_prompt(self.prompt_name)
        msg = prompt.compile(abstract=abstract_text)
        response = self.get_llm_response(msg)
        result = {
            'Abstract Number': abstract_number,
            'Abstract Text': abstract_text,
            'response': response,
            **self.extract_info(response)
        }
        
        with open(temp_file, 'w') as f:
            json.dump(result, f)
        
        return result

class IndicationExtractor(Extractor):
    def __init__(self, model='groq'):
        super().__init__('indication', 'GetIndication', ['indication', 'subtype'], model)

class PhaseExtractor(Extractor):
    def __init__(self, model='groq'):
        super().__init__('phase', 'Phase', ['phase', 'model'], model)

class InitialGeneExtractor(Extractor):
    def __init__(self, model='groq'):
        super().__init__('initial_gene', 'all_genes', ['reasoning', 'genes'], model)


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

    def process_abstract(self, abstract_number, abstract_text, initial_result):
        temp_file = os.path.join(self.temp_folder, f"{abstract_number}.json")
        
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                return json.load(f)
        
        potential_genes = initial_result.get('genes', [])
        
        max_retries = 5
        gene_context = []
        symbols_only = []
        for attempt in range(max_retries):
            try:
                gene_context, symbols_only = self.get_gene_context(potential_genes)
                break  
            except Exception as e:
                print(f"Error occurred: {e}. Retrying in 20 seconds...")
                time.sleep(20)

        prompt = self.langfuse.get_prompt(self.prompt_name)
        msg = prompt.compile(abstract=abstract_text, potential_genes=potential_genes, gene_context=gene_context, symbols_only=symbols_only)

        response = self.get_llm_response(msg)
        
        result = {
            'Abstract Number': abstract_number,
            'Abstract Text': abstract_text,
            'potential_genes': potential_genes,
            'initial_reasoning': initial_result.get('reasoning', ''),
            'gene_context': gene_context,
            'symbols_only': symbols_only,
            **self.extract_info(response)
        }
        
        with open(temp_file, 'w') as f:
            json.dump(result, f)
        
        return result