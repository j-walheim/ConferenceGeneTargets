# %%
import os
import requests
from llama_parse import LlamaParse
from llama_index.core import Document
import pickle
from PyPDF2 import PdfReader, PdfWriter
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# %%

# Download PDF
proceedings_url = 'https://www.aacr.org/wp-content/uploads/2024/04/AACR2024_Regular_Abstracts_04-01-24.pdf'
pdf_path = "data/pdfs/"
fname = os.path.join(pdf_path, os.path.basename(proceedings_url))

if not os.path.exists(fname):
    os.makedirs(pdf_path, exist_ok=True)
    response = requests.get(proceedings_url)
    response.raise_for_status()
    with open(fname, 'wb') as file:
        file.write(response.content)
    print(f"PDF downloaded to {fname}")
else:
    print("PDF already downloaded.")

# Initialize LlamaParse
parser = LlamaParse(
    api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
    result_type="markdown"
)

# Create output directories
pages_dir = "data/processed_pages"
os.makedirs(pages_dir, exist_ok=True)

# Function to process a single page
def process_page(page_content, page_number):
    # Create a temporary file for the single page
    temp_file = f"temp_page_{page_number}.pdf"
    writer = PdfWriter()
    writer.add_page(page_content)
    with open(temp_file, "wb") as output_file:
        writer.write(output_file)
    
    # Process the page
    documents = parser.load_data(temp_file)
    
    # Remove the temporary file
    os.remove(temp_file)
    
    # Add page number to metadata
    for doc in documents:
        doc.metadata['page_number'] = page_number
    
    return documents

# Read and process the PDF
reader = PdfReader(fname)
all_documents = []

# Use tqdm for a progress bar
for i, page in enumerate(tqdm(reader.pages, desc="Processing pages")):
    page_docs = process_page(page, i + 1)
    all_documents.extend(page_docs)
    
    # Save each page separately
    page_file = os.path.join(pages_dir, f"page_{i+1}.pkl")
    with open(page_file, "wb") as f:
        pickle.dump(page_docs, f)


# Store combined results as pickle
with open("data/all_abstracts.pkl", "wb") as f:
    pickle.dump(all_documents, f)

print(f"All {len(all_documents)} pages processed and saved.")
print(f"Individual pages saved in {pages_dir}")


# %%
# convert to JSON

# read in the pickle file and convert to JSON
with open("data/processed_pages/page_10.pkl", "rb") as f:
    data = pickle.load(f)




# %%
data[0].text
# %%
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

with open('../.keys/mistral', 'r') as file:
    os.environ['MISTRAL_API_KEY'] = file.read().strip()
    
client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"))
model = 'open-mistral-7b'

with open('prompts/prompt_naive', 'r') as f:
    prompt = f.read()

abstracts_dir = 'data/processed_airflow/processed_pages/'

out_json = 'data/processed_airflow/processed_json/'

# add directory to out_json if it doesn't exist
os.makedirs(out_json, exist_ok=True)

fname_abstract = "data/processed_pages/page_10.pkl"

fname_out = os.path.join(out_json, os.path.basename(fname_abstract))


# read in the pickle file and convert to JSON
with open(fname_abstract, "rb") as f:
    data = pickle.load(f)
    
# get text 
abstract = data[0].text

prompt_cur = prompt
prompt_cur = prompt_cur.replace('[[[abstract]]]', abstract)

# %%
# List to store the results
results = []
# Open the input CSV file
# Query the model and store the response
chat_response = client.chat(
    model=model,
    messages=[ChatMessage(role="user", content=prompt_cur)]
)
result = chat_response.choices[0].message.content

# %%
# Try to parse the result as JSON
try:
    response_json = json.loads(result)
    results.append(response_json)
    
    # Save to JSON file
    output_file = 'results/output.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    
    output_file = 'results/output.txt'
    with open(output_file, 'w') as f:
        f.write(result)
    


with open(abstracts_file, "r") as csv_file:
    reader = csv.DictReader(csv_file)
    # Iterate through the rows in the input CSV file
    iteration_counter = 0
    for row in tqdm(reader, desc="Processing rows"):
        iteration_counter += 1
        if iteration_counter >= 10:
             break
        prompt_cur = prompt
        prompt_cur = prompt_cur.replace('[[[abstract]]]', row['Abstract'])
        try:
            # Query the model and store the response
            chat_response = client.chat(
                model=model,
                messages=[ChatMessage(role="user", content=prompt_cur)]
            )
            row['response'] = chat_response.choices[0].message.content
        except Exception as e:
            print(f"Error querying model: {e}")
            row['response'] = "Error querying model"
        results.append(row)
        
        # Save intermediate results every 100 iterations
        if iteration_counter % 50 == 0:
            with open('intermediate_results.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"Saved intermediate results at iteration {iteration_counter}")
                
#export results as pickle - sometimes json parsing fails, don't want to lose all results
with open('results/tmp.pkl', 'w') as f:
    json.dump(results, f, indent=4)


# convert results to dataframe
results = pd.DataFrame(results)

results_json = []

# Iterate over DataFrame rows using iterrows()
for _, row in tqdm(results.iterrows(), total=len(results), desc="Producing JSON"):
    try:
        response_json = json.loads(row['response'])
        response_json['abstract'] = row['Abstract']
        results_json.append(response_json)
    except Exception as e:
        print(f"Error parsing JSON: {e}")

# Export results
with open(out_file_json, "w") as json_file:
    json.dump(results_json, json_file, indent=4)

with open('results/last_results.json', "w") as json_file:
    json.dump(results_json, json_file, indent=4)
    


