import os
import json
import pandas as pd
from glob import glob

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
    return pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)

# Load JSON files from data/temporary_indication
indication_df = load_json_files('data/temporary_indication')

# Load JSON files from data/temporary_stage
stage_df = load_json_files('data/temporary_stage')

# Check if dataframes are empty
if indication_df.empty:
    print("Warning: indication_df is empty")
if stage_df.empty:
    print("Warning: stage_df is empty")

# Print column names for debugging
print("Indication columns:", indication_df.columns)
print("Stage columns:", stage_df.columns)

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

# Merge with CSV data
if 'Abstract Number' in merged_df.columns and 'Abstract Number' in csv_df.columns:
    final_df = pd.merge(merged_df, csv_df, on='Abstract Number', how='outer')
else:
    print("Error: 'Abstract Number' column not found for merging with CSV data")
    final_df = pd.concat([merged_df, csv_df], axis=1)

#drop missing indication
final_df = final_df.dropna(subset=['Extracted Indication','reasoning_second_prompt'])

# Display the first few rows of the final merged dataframe
print(final_df.head())

# Save the final merged dataframe to a CSV file
#final_df.to_csv('final_merged_data.csv', index=False)
# save as xlsx
final_df.to_excel('final_merged_data.xlsx', index=False)

# save to jsonl
with open('final_merged_data.jsonl', 'w') as f:
    for _, row in final_df.iterrows():
        f.write(json.dumps(row.to_dict()) + '\n')

print(f"Final dataframe shape: {final_df.shape}")
print(f"Final dataframe columns: {final_df.columns}")


#