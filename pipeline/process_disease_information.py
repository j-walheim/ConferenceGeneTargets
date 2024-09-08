from .utils import get_llm, create_extraction_chain, initialize_vectorstore
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from defs.abstract_class import Disease


def create_disease_prompt(abstract_text, disease_context):

    ##### Prompt element 1: `user` role
    # Make sure that your Messages API call always starts with a `user` role in the messages array.
    # The get_completion() function as defined above will automatically do this for you.

    ##### Prompt element 2: Task context
    # Give context about the role it should take on or what goals and overarching tasks you want it to undertake with the prompt.
    # It's best to put context early in the body of the prompt.
    TASK_CONTEXT = "You are an oncologist and an expert at extracting information from academic abstracts."
    

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
    EXAMPLES = """ Focus on the main disease mentioned in the abstract.

    <examples>
    <example>
    These data lay the foundation for first-in-human clinical translation of a T-cell based therapy targeting Ewing sarcoma. // Disease: Ewing sarcoma
    </example>
    <example>
    We screened compounds for their ability to reduce CD8+ T cell senescence in various cancer types. // Disease: Cancer (general)
    </example>
    <example>
    Our methodology allows to cultivate better CD8+ T cells. // Disease: None
    </example>
    <example>
    This research investigates the efficacy of immunotherapy in triple-negative breast cancer (TNBC). // Disease: Triple-negative breast cancer
    </example>
    
    </examples>"""

    ##### Prompt element 6: Detailed task description and rules
    # Expand on the specific tasks you want LLM to do, as well as any rules that LLM might have to follow.
    # This is also where you can give LLM an "out" if it doesn't have an answer or doesn't know.
    # It's ideal to show this description and rules to a friend to make sure it is laid out logically and that any ambiguous words are clearly defined.
    TASK_DESCRIPTION = """
    
    Your task is to accurately extract the primary disease or cancer type that is explicitly stated as the main focus of the research or treatment in the abstract. 
    This should be a specific type of cancer or disease. Be as specific as possible, if general cancer is mentioned, use "Cancer (general)". 
    If multiple diseases are mentioned, list all of them.

    Follow these guidelines strictly:
    - NEVER GUESS OR INFER INFORMATION THAT IS NOT EXPLICITLY STATED IN THE ABSTRACT. Accuracy is paramount. It is better to omit information than to provide potentially incorrect data.
    - If no specific disease or cancer type is mentioned, or if it's a general study about cancer, use "Cancer (general)".
    - If no disease is mentioned (e.g. abstract about a laboratory technique), use "None".
    - Do not attempt to fill in missing information based on context or general knowledge.

    <abstract_text>
    {abstract_text}
    </abstract_text>

    All extracted diseases must be part of this context:
    <disease_context>
    {disease_context}
    </disease_context>
    
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
    PRECOGNITION = ""

    ##### Prompt element 9: Output formatting
    # If there is a specific way you want LLM's response formatted, clearly tell LLM what that format is.
    # This element may not be necessary depending on the task.
    # If you include it, putting it toward the end of the prompt is better than at the beginning.
    OUTPUT_FORMATTING = "ONLY output to disease name or names in <answer> tags."

    ##### Prompt element 10: Prefilling LLM's response (if any)
    # A space to start off LLM's answer with some prefilled words to steer LLM's behavior or response.
    # If you want to prefill LLM's response, you must put this in the `assistant` role in the API call.
    # This element may not be necessary depending on the task.
    PREFILL = ""


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

def extract_disease_info(abstract_text, model='groq', vectorstore=None):
    llm = get_llm(model)
    disease_context = vectorstore.rag('diseases', abstract_text) if vectorstore else ""
    print(disease_context)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=create_disease_prompt(abstract_text, disease_context))
    ])

    extraction_chain = create_extraction_chain(prompt, llm, Disease)
    result = extraction_chain.invoke({"text": abstract_text})
    return result

# Example usage
if __name__ == "__main__":
    abstract_text = "Your abstract text here..."
    result = extract_disease_info(abstract_text)
    print(f"Extracted disease: {result.disease}")


