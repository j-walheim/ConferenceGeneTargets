import os
import csv
import json
from glob import glob

# Define the input and output directories
input_dir = 'data/results_posters/production/parsed_pages_gpt-4o'
output_dir = 'data/results_posters/temporary_gene_target_old'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get a list of all CSV files in the input directory
csv_files = glob(os.path.join(input_dir, '*.csv'))

# Define field mapping
field_mapping = {
    'reasoning_second_prompt': 'reasoning',
    'target': 'gene_target'
}

# Process each CSV file
for csv_file in csv_files:
    print(f"Processing {csv_file}")
    # Read the CSV file
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)  # Get the first (and only) row
        new_row = {}
        for key, value in row.items():
            new_key = field_mapping.get(key, key)
            new_row[new_key] = value
    
    # Convert the single row to JSON format
    json_data = json.dumps(new_row, indent=4)
    
    # Define the output JSON file path
    base_name = os.path.basename(csv_file)
    json_file = os.path.join(output_dir, f'{os.path.splitext(base_name)[0]}.json')
    
    # Write the JSON data to the output file
    with open(json_file, 'w') as f:
        f.write(json_data)


print(f"Conversion completed. JSON files have been saved to {output_dir}.")
