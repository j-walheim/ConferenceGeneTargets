import os
import json
import pandas as pd
from tqdm import tqdm

# Define the directory path
dir_path = '/teamspace/studios/this_studio/ConferenceGeneTargets/data/production/processed_pages'

# Initialize an empty list to store all abstracts
all_abstracts = []

# Get the list of JSON files
json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]

# Iterate through all JSON files in the directory with tqdm progress bar
for filename in tqdm(json_files, desc="Processing JSON files"):
    file_path = os.path.join(dir_path, filename)
    with open(file_path, 'r') as file:
        data = json.load(file)
        for item in data:
            all_abstracts.append({
                'page_number': item['page_number'],
                'content': item['content']
            })

# Create a DataFrame from the list of abstracts
df_abstracts = pd.DataFrame(all_abstracts)

# Sort the DataFrame by page number if needed
df_abstracts = df_abstracts.sort_values('page_number')

# Reset the index
df_abstracts = df_abstracts.reset_index(drop=True)

# Now df_abstracts contains all the abstracts from all JSON files
print(f"Total number of abstracts loaded: {len(df_abstracts)}")
print(df_abstracts.head())

# save as csv
df_abstracts.to_csv('data/production/abstracts.csv', index=False)