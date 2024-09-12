import os
import json
import pandas as pd
from glob import glob
import csv

def load_json_files(directory):
    data_list = []
    for json_file in glob(os.path.join(directory, '*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            data_list.append(data)
        elif isinstance(data, list):
            data_list.extend(data)
        else:
            print(f"Unexpected data type in {json_file}: {type(data)}")
    return pd.DataFrame(data_list)

def load_csv_files(directory):
    all_csvs = glob(os.path.join(directory, '*.csv'))
    dataframes = []
    for csv_file in all_csvs:
        try:
            df = pd.read_csv(csv_file, quoting=csv.QUOTE_ALL, escapechar='\\')
            dataframes.append(df)
        except pd.errors.ParserError as e:
            print(f"Error reading {csv_file}: {e}")
            # Attempt to read the file with a different encoding or quoting method
            try:
                df = pd.read_csv(csv_file, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
                dataframes.append(df)
            except Exception as e2:
                print(f"Failed to read {csv_file} with alternative method: {e2}")
    
    if not dataframes:
        print("No valid CSV files found.")
        return pd.DataFrame()
    
    return pd.concat(dataframes, ignore_index=True)
# Load JSON files from data/temporary_indication
indication_df = load_json_files('data/temporary_indication_gpt4o')
indication_df = indication_df.drop(columns=['Abstract Text'])
# Load JSON files from data/temporary_stage
stage_df = load_json_files('data/temporary_phase')

# Check if dataframes are empty
if indication_df.empty:
    print("Warning: indication_df is empty")
if stage_df.empty:
    print("Warning: stage_df is empty")

# Print column names for debugging
print("Indication columns:", indication_df.columns)
print("Stage columns:", stage_df.columns)

# store both dataframes in data folder
indication_df.to_csv('indication_df.csv', index=False)
stage_df.to_csv('stage_df.csv', index=False)

# Merge dataframes by "Abstract Number" if both dataframes have this column
if 'Abstract Number' in indication_df.columns and 'Abstract Number' in stage_df.columns:
    merged_df = pd.merge(indication_df, stage_df, on='Abstract Number', how='outer')
else:
    print("Error: 'Abstract Number' column not found in one or both dataframes")
    merged_df = pd.concat([indication_df, stage_df], axis=1)

# Load CSV files from data/production/parsed_pages_gpt-4o
csv_df = load_csv_files('data/production/parsed_pages_gpt-4o')
# rename page_number to Abstract Number
csv_df = csv_df.rename(columns={'page_number': 'Abstract Number'})

# Print CSV columns for debugging
print("CSV columns:", csv_df.columns)


csv_df = csv_df.drop(columns=['text'])
# Merge with CSV data
if 'Abstract Number' in merged_df.columns and 'Abstract Number' in csv_df.columns:
    final_df = pd.merge(merged_df, csv_df, on='Abstract Number', how='left')
else:
    print("Error: 'Abstract Number' column not found for merging with CSV data")
    final_df = pd.concat([merged_df, csv_df], axis=1)

#drop missing indication
#final_df = final_df.dropna(subset=['Extracted Indication','reasoning_second_prompt'])

# Display the first few rows of the final merged dataframe
print(final_df.head())

# Create the results directory if it doesn't exist
results_dir = 'data/results'
os.makedirs(results_dir, exist_ok=True)

# Save the final merged dataframe to an Excel file
final_df.to_excel(os.path.join(results_dir, 'final_merged_data.xlsx'), index=False)

# Save to jsonl
with open(os.path.join(results_dir, 'final_merged_data.jsonl'), 'w') as f:
    for _, row in final_df.iterrows():
        f.write(json.dumps(row.to_dict()) + '\n')

print(f"Final dataframe shape: {final_df.shape}")
print(f"Final dataframe columns: {final_df.columns}")
