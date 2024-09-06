from .utils import get_llm, create_extraction_chain, vectorstore
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from defs.abstract_class import Target

def create_prompt(abstract_text, gene_context):

    ##### Prompt element 1: `user` role
    # Make sure that your Messages API call always starts with a `user` role in the messages array.
    # The get_completion() function as defined above will automatically do this for you.

    ##### Prompt element 2: Task context
    # Give context about the role it should take on or what goals and overarching tasks you want it to undertake with the prompt.
    # It's best to put context early in the body of the prompt.
    TASK_CONTEXT = "You are an expert at extracting information from academic abstracts, with a focus on drug targets, genetics and oncology."

    ##### Prompt element 3: Tone context
    # If important to the interaction, tell LLM what tone it should use.
    # This element may not be necessary depending on the task.
    TONE_CONTEXT = ""

    ##### Prompt element 4: Input data to process
    # If there is data that LLM needs to process within the prompt, include it here within relevant XML tags.
    # Feel free to include multiple pieces of data, but be sure to enclose each in its own set of XML tags.
    # This element may not be necessary depending on task. Ordering is also flexible.
    INPUT_DATA = f"""This is the abstract text. Only extract information from here.
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
    EXAMPLES = """Make sure to focus on the target that is or can be modulated. Ignore biomarkers and/or pathways with predictive or prognostic value. Ignore any gene that is not directly modulated or can be modulated to change the disease.

    <examples>
    <example>
    These data lay the foundation for first-in-human clinical translation of a T-cell based therapy targeting EWSR1-WT1 // Target: EWSR1-WT1
    </example>
    <example>
    Reactive T cells were identified upon detection of secreted cytokines (IFNγ, TNFα, IL2) and measurement of 4-1BB (CD137) surface expression. // Target: None
    </example>
    <example>
    We use the chimeric costimulatory switch protein (CSP) PD1-41BB that turns the inhibitory signal mediated via the PD-1-PD-L1 axis into a costimulatory one. // Target: PD1-41BB
    </example>
    <example>
    We screened ginsenoside Rb1 from 22 anti-aging compounds as the compound that had the most effective activity to reduce CD8+ T cell senescence. // Target: None
    </example>
    <example>
    This PDF contains all regular abstracts (submitted for the November 16, 2023 deadline) that were scheduled for presentation as of Monday, April 1, 2024. // Target: None
    </example>
    <example>
    PD-L1 is able to stratify patients into responders vs. non-responders to immunotherapy. // Target: None
    </example>
    <example>
    KRAS expression alone has limited predictive value for immunotherapy in TNBC. // Target: None
    <example>
    EGFR is a prognostic biomarker. // Target: None
    </example>
    <example>
    We found novel drugs predicted against Ephrin-Type-A Receptor 2, chosen through preliminary literature and genomic analysis. // Target: EPHA2
    </example>
    <example>
    BRAF melanoma cancers. // Target: None
    </example>
    </examples>"""

    ##### Prompt element 6: Detailed task description and rules
    # Expand on the specific tasks you want LLM to do, as well as any rules that LLM might have to follow.
    # This is also where you can give LLM an "out" if it doesn't have an answer or doesn't know.
    # It's ideal to show this description and rules to a friend to make sure it is laid out logically and that any ambiguous words are clearly defined.
    TASK_DESCRIPTION = """
    Your task is to extract Find the primary gene target that is expicitely stated to impact the disease and can be modulated to change the disease or is used to selectively 
    and specifically target the disease in the abstract. If a pathway is mentioned, try to get the key gene in the pathway. Modulation can include inhibition, activation, knocked-in, 
    knocked-out, or any other type of modulation. 
    It can be stated that the drug is targeting, acting on, acting against, having high selectivity, having avidity, binding to a certain target.
    - NEVER GUESS OR INFER INFORMATION THAT IS NOT EXPILICTLY STATED IN THE ABSTRACT. ONLY MAP TO SYNONYMS.
    - Accuracy is paramount. It is better to omit information than to provide potentially incorrect data. Under no circumstances should you guess or make assumptions about any data.
    - Do not attempt to fill in missing information based on context or general knowledge.
        
    If a target is mentioned in the abstract, extract it. 
    If protein names are mentioned, use the gene symbol (e.g. TP53 for p53). The target could be a specific mutation or version of a gene, e.g. KRASG12D is a common mutation for KRAS. Extract KRAS in this case.
    Ignore biomarkers and/or pathways with predictive or prognostic value. Ignore any gene that is not directly modulated or can be modulated to change the disease.


    <abstract_text>
    {abstract_text}
    </abstract_text>

    All extracted genes must be part of this context:
    <gene_context>
    {gene_context}
    </gene_context>
    
    """

    ##### Prompt element 7: Immediate task description or request #####
    # "Remind" LLM or tell LLM exactly what it's expected to immediately do to fulfill the prompt's task.
    # This is also where you would put in additional variables like the user's question.
    # It generally doesn't hurt to reiterate to LLM its immediate task. It's best to do this toward the end of a long prompt.
    # This will yield better results than putting this at the beginning.
    # It is also generally good practice to put the user's query close to the bottom of the prompt.
    IMMEDIATE_TASK = ""

    ##### Prompt element 8: Precognition (thinking step by step)
    # For tasks with multiple steps, it's good to tell LLM to think step by step before giving an answer
    # Sometimes, you might have to even say "Before you give your answer..." just to make sure LLM does this first.
    # Not necessary with all prompts, though if included, it's best to do this toward the end of a long prompt and right after the final immediate task request or description.
    # PRECOGNITION = "Before you answer, pull out the most relevant quotes from the research in <relevant_quotes> tags."
    PRECOGNITION = "Before you answer, make sure that the targets are not biomarkers or refering to characteristics of the disease, rather than drug targets."

    ##### Prompt element 9: Output formatting
    # If there is a specific way you want LLM's response formatted, clearly tell LLM what that format is.
    # This element may not be necessary depending on the task.
    # If you include it, putting it toward the end of the prompt is better than at the beginning.
    OUTPUT_FORMATTING = "ONLY output to gene name or gene names in <answer> tags."

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
    gene_context = vectorstore.rag('gene', abstract_text) if vectorstore else ""
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=create_prompt(abstract_text, gene_context))
    ])

    extraction_chain = create_extraction_chain(prompt, llm, Target)
    result = extraction_chain.invoke({"text": abstract_text})
    return result

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