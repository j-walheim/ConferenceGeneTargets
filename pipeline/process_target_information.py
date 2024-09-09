from .utils import get_llm, create_extraction_chain
from pydantic import BaseModel, Field

def create_initial_prompt_all_genes(abstract_text):
    return f"""
    You are an expert at extracting gene or protein names from academic abstracts.
    This is the abstract text. Only extract information from here.
    <abstract_text>
    {abstract_text}
    </abstract_text>
    
    Make sure to focus on ALL genes or proteins that are mentioned in the abstract and in every variant of how they appear, regardless of whether they are biomarkers or drug targets themselves.
    If a gene is mentioned in the abstract, extract it. Extract all genes separately, if a fusion is mentioned, extract the two genes separately.

    <examples>
    <example>
    PD-L1 is able to stratify patients responding to LAG-3 therapy. // Target: PD-L1, LAG-3
    </example>
    <example>
    Epidermal growth factor receptor is commonly mutated in lung cancer. // Target: Epidermal growth factor receptor
    </example>
    <example>
    Receptor tyrosine-protein kinase erbB-2 is a known biomarker. HER2 is the protein coded by gene ERBB2. // Target: Receptor tyrosine-protein kinase erbB-2, HER2, ERBB2
    </example>
    <example>
    Inhibiting B-cells shows beneficial effects in PDAC. // Target: 
    </example>
    <example>
    SDJS, a novel cell type identified by FACS, can be activated against cancer cells. // Target: 
    </example>
    <example>
    Mice have been injected with 1000 ml NSCLC S23 cells. // Target: 
    </example>
    <example>
    Azanobitin is a potent modulator of tumor  microenvironment in bladder cell lines. // Target: 
    </example>
    <example>
    Degradation of Pdcd4 enhances enzalutamide resistance in prostate cancer. // Target: Pdcd4
    </example>
    <example>
    High levels of WRP are indicative of advanced disease in breast cancer. // Target: WRP
    </example>
    </examples>
    Your task is to retrieve all gene names, gene symbols or protein names that are mentioned in the abstract.
    
    Rules: 
    - Accuracy is paramount. It is better to omit information than to provide potentially incorrect data. Under no circumstances should you guess or make assumptions about any data.
    - Do not attempt to fill in missing information based on context or general knowledge.
    - If you are very confident about a gene synonym, take the official HUGO symbol. (e.g. ERBB2 is the official HUGO symbol for HER2)
    
    Before extracting any gene, ask yourself "is this a gene or a protein?" If the answer is yes, include it. Otherwise, exclude it. DO NOT INCLUDE DRUG NAMES.

    Write your reasoning for classifying and selecting the entries as genes in <reasoning> </reasoning> tags.

    Output to gene name or gene names in <answer> </answer> tags
    """


def create_prompt(abstract_text, potential_genes, gene_context, symbols_only):

    ##### Prompt element 1: `user` role
    # Make sure that your Messages API call always starts with a `user` role in the messages array.
    # The get_completion() function as defined above will automatically do this for you.

    ##### Prompt element 2: Task context
    # Give context about the role it should take on or what goals and overarching tasks you want it to undertake with the prompt.
    # It's best to put context early in the body of the prompt.
    TASK_CONTEXT = "You are an expert at extracting the most relevant drug targets from academic abstracts."

    ##### Prompt element 3: Tone context
    # If important to the interaction, tell LLM what tone it should use.
    # This element may not be necessary depending on the task.
    TONE_CONTEXT = ""

    ##### Prompt element 4: Input data to process
    # If there is data that LLM needs to process within the prompt, include it here within relevant XML tags.
    # Feel free to include multiple pieces of data, but be sure to enclose each in its own set of XML tags.
    # This element may not be necessary depending on task. Ordering is also flexible.
    INPUT_DATA = f"""Only extract information from this abstract text. These are all the genes mentioned in the abstract: {potential_genes}. Your task is to FILTER OUT the genes that are not a drug target in the study,
    remove all gene names that are either biomarkes (e.g. mutated genes, readout genes, biomarkers for predictive or prognostic value) or are not the primary focus of the study (e.g. for comparison or background information).
    This is the abstract text:
    <abstract_text>
    {abstract_text}
    </abstract_text>
    """

    ##### Prompt element 5: Examples
    # Provide LLM with at least one example of an ideal response that it can emulate. Encase this in <example></example> XML tags. Feel free to provide multiple examples.
    # If you do provide multiple examples, give LLM context about what it is an example of, and enclose each example in its own set of XML tags.
    # Examples are probably the single most effective tool in knowledge work for getting LLM to behave as desired.
    # Make sure to give LLM examples of common edge cases. If your prompt uses a scratchpad, it's effective to give examples of how the scratchpad should look.
    # Generally more examples = better.
    EXAMPLES = """Make sure to focus on the targets that are or can be directly modulated. Focus only on gene targets, ignore cell lines, drugs and anything else. Ignore biomarkers and/or pathways with predictive or prognostic value. Ignore any gene that is not directly modulated or can be modulated to change the disease.

    <examples>
    <example>
    These data lay the foundation for first-in-human clinical translation of a T-cell based therapy targeting EWSR1-WT1 // Target: EWSR1-WT1
    </example>
    <example>
    Reactive T cells were identified upon detection of secreted cytokines (IFNγ, TNFα, IL2) and measurement of 4-1BB (CD137) surface expression. // Target:
    </example>
    <example>
    We use the chimeric costimulatory switch protein (CSP) PD1-41BB that turns the inhibitory signal mediated via the PD-1-PD-L1 axis into a costimulatory one. // Target: PD1-41BB
    </example>
    <example>
    We screened ginsenoside Rb1 from 22 anti-aging compounds as the compound that had the most effective activity to reduce CD8+ T cell senescence. // Target:
    </example>
    <example>
    This PDF contains all regular abstracts (submitted for the November 16, 2023 deadline) that were scheduled for presentation as of Monday, April 1, 2024. // Target:
    </example>
    <example>
    PD-L1 is able to stratify patients into responders vs. non-responders to immunotherapy. // Target:
    </example>
    <example>
    Our compound shows better efficacy in preclinical models as compared to PARP inhibitors. // Target:
    </example>
    <example>
    KRAS expression alone has limited predictive value for immunotherapy in TNBC. // Target:
    <example>
    EGFR is a prognostic biomarker. // Target:
    </example>
    <example>
    We found novel drugs predicted against Ephrin-Type-A Receptor 2, chosen through preliminary literature and genomic analysis. // Target: EPHA2
    </example>
    <example>
    BRAF melanoma cancers. // Target:
    </example>
    <example>
    The inhibition of IL2, and the activation of IL13. // Target: IL2, IL13
    </example>
    <example>
    MTPNR mutated patients show a better ORR in response to immunotherapy. // Target:
    </example>
    <example>
    This PSAUR-directed antibody has far reaching potential in all solid turmos. // Target: PSAUR
    </example>
    </examples>"""

    ##### Prompt element 6: Detailed task description and rules
    # Expand on the specific tasks you want LLM to do, as well as any rules that LLM might have to follow.
    # This is also where you can give LLM an "out" if it doesn't have an answer or doesn't know.
    # It's ideal to show this description and rules to a friend to make sure it is laid out logically and that any ambiguous words are clearly defined.
    TASK_DESCRIPTION = f"""
    Your task is to only keep the primary gene targets that are expicitely stated to impact the disease and can be modulated to change the disease or is used to selectively 
    and specifically target the disease in the abstract. If a pathway is mentioned, try to get the key gene in the pathway. Modulation can include inhibition, activation, knocked-in, 
    knocked-out, or any other type of modulation. 
    It can be stated that the drug is targeting, acting on, acting against, having high selectivity, having avidity, binding to a certain target.
    - NEVER GUESS OR INFER INFORMATION THAT IS NOT EXPILICTLY STATED IN THE ABSTRACT. ONLY MAP TO SYNONYMS.
    - Accuracy is paramount. It is better to omit information than to provide potentially incorrect data. Under no circumstances should you guess or make assumptions about any data.
    - Do not attempt to fill in missing information based on context or general knowledge.
    
    If protein names are mentioned, use the gene symbol (e.g. TP53 for p53). The target could be a specific mutation or version of a gene, e.g. KRASG12D is a common mutation for KRAS. Extract KRAS in this case.
    Ignore predictive or progrostic genes/biomarkers and/or pathways. Ignore any gene that is not directly modulated or can be modulated to change the disease.

    We want to standardize the drug targets names to HUGO gene nomenclature. To faciliate this you can refer to the following relevant context: 
    <gene_context> 
    {gene_context} 
    </gene_context>

    Try to match genes to the following list, if you do not find a good mapping <synonym_list>{symbols_only}</synonym_list>, keep the original name.
    """

    # All extracted genes must be part of this context. If nothing is provided as context, do not include the gene. Translate the gene names to synonyms if possible.
    
    # <gene_context> 
    # {gene_context} 
    # </gene_context>

    ##### Prompt element 7: Immediate task description or request #####
    # "Remind" LLM or tell LLM exactly what it's expected to immediately do to fulfill the prompt's task.
    # This is also where you would put in additional variables like the user's question.
    # It generally doesn't hurt to reiterate to LLM its immediate task. It's best to do this toward the end of a long prompt.
    # It is also generally good practice to put the user's query close to the bottom of the prompt.
    IMMEDIATE_TASK = ""

    ##### Prompt element 8: Precognition (thinking step by step)
    # For tasks with multiple steps, it's good to tell LLM to think step by step before giving an answer
    # Sometimes, you might have to even say "Before you give your answer..." just to make sure LLM does this first.
    # Not necessary with all prompts, though if included, it's best to do this toward the end of a long prompt and right after the final immediate task request or description.
    # PRECOGNITION = "Before you answer, pull out the most relevant quotes from the research in <relevant_quotes> tags."
    PRECOGNITION = "Before you answer, ask yourself 'is this a gene or protein that is modulated to change the disease?' Ignore any gene that is not directly modulated, but is only reported as biomarker or signal for a biological process."

    ##### Prompt element 9: Output formatting
    # If there is a specific way you want LLM's response formatted, clearly tell LLM what that format is.
    # This element may not be necessary depending on the task.
    # If you include it, putting it toward the end of the prompt is better than at the beginning.
    OUTPUT_FORMATTING = """
    
    First, write your reasoning for classifying and selecting the entries as targets in <reasoning> </reasoning> tags.

    Then, output to gene name or gene names in <answer> </answer> tags. If no gene is a drug target, output <answer></answer>.
    """

    ##### Prompt element 10: Prefilling LLM's response (if any)
    # A space to start off LLM's answer with some prefilled words to steer LLM's behavior or response.
    # If you want to prefill LLM's response, you must put this in the `assistant` role in the API call.
    # This element may not be necessary depending on the task.
    PREFILL = "<relevant_quotes>"


    ######################################## COMBINE ELEMENTS ########################################

    PROMPT = ""

    if TASK_CONTEXT:
        PROMPT += f"""{TASK_CONTEXT}"""

    if TONE_CONTEXT:
        PROMPT += f"""\n\n{TONE_CONTEXT}"""

    if INPUT_DATA:
        PROMPT += f"""\n\n{INPUT_DATA}"""

    if EXAMPLES:
        PROMPT += f"""\n\n{EXAMPLES}"""

    if TASK_DESCRIPTION:
        PROMPT += f"""\n\n{TASK_DESCRIPTION}"""

    if IMMEDIATE_TASK:
        PROMPT += f"""\n\n{IMMEDIATE_TASK}"""

    if PRECOGNITION:
        PROMPT += f"""\n\n{PRECOGNITION}"""

    if OUTPUT_FORMATTING:
        PROMPT += f"""\n\n{OUTPUT_FORMATTING}"""


    return PROMPT

def extract_target_from_abstract(abstract_text, model='groq', vectorstore=None):
    llm = get_llm(model)
    # First, extract potential genes from the abstract
    initial_prompt = create_initial_prompt_all_genes(abstract_text)
    results_first_prompt = create_extraction_chain(initial_prompt, llm)
    print("results_first_prompt: ", results_first_prompt)
    
    potential_genes = []
    reasoning = ""
    if isinstance(results_first_prompt, dict):
        potential_genes = results_first_prompt.get('extracted', [])
        reasoning = results_first_prompt.get('reasoning', [""])[0]
    else:
        print(f"Unexpected type for results_first_prompt: {type(results_first_prompt)}")
    
    print("Potential genes:", potential_genes)
    
    # Get context for potential genes using vectorstore
    gene_context = [""]
    symbols_only = [""]
    if vectorstore and len(potential_genes) > 0:
        for gene in potential_genes:
            context = vectorstore.retrieve('gene-index', gene)
            # Symbols only
            symbols_only = [item['symbol'] for item in context]

            gene_context.append(f"{gene}: {context}")
    
    print("gene_context: ", gene_context)

    # Create the final prompt with the abstract and gene context
    second_prompt = create_prompt(abstract_text, potential_genes, gene_context, symbols_only)
    result_second_prompt = create_extraction_chain(second_prompt, llm)

    reasoning_second_prompt = result_second_prompt.get('reasoning', [""])[0]
    targets = result_second_prompt.get('extracted', [])
    
    return reasoning, potential_genes, gene_context, symbols_only, reasoning_second_prompt, targets

#@task
def create_jsonl_from_parsed_pages(**kwargs):
#    ti = kwargs['ti']
    model = 'groq'
    parsed_pages_dir = os.path.join(STORAGE_DIR, ENVIRONMENT, f'parsed_pages_{model}') 
    output_jsonl = os.path.join(STORAGE_DIR, ENVIRONMENT, f'parsed_pages_{model}.jsonl')
    with open(output_jsonl, 'w') as jsonl_file:
        for filename in sorted(glob.glob(os.path.join(parsed_pages_dir, 'page_*.json'))):
            with open(filename, 'r') as json_file:
                abstract_dict = json.load(json_file)
                json.dump(abstract_dict, jsonl_file)
                jsonl_file.write('\n')