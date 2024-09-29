import duckdb
import pandas as pd
import re

# Connect to the DuckDB database
conn = duckdb.connect('conference_gene_targets.db')

# Step 1: Define cleaning functions
def extract_abstract_title(text):
    if not isinstance(text, str):
        return ""
    match = re.search(r'-\s*(.*?)\s*Background', text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

def handle_her2_erbb2(genes):
    gene_list = eval(genes)
    if "HER2" in gene_list and "ERBB2" not in gene_list:
        gene_list = [gene.replace("HER2", "ERBB2") for gene in gene_list]
    elif "HER2" in gene_list and "ERBB2" in gene_list:
        gene_list = [gene for gene in gene_list if gene != "HER2"]
    return str(gene_list)

def standardize_phase(phase):
    if not isinstance(phase, str):
        return "Unknown"
    phase = phase.title()
    roman_to_arabic = {'I': '1', 'Ii': '2', 'Iii': '3', 'Iv': '4', 'V': '5'}
    for roman, arabic in roman_to_arabic.items():
        phase = re.sub(rf'\b{roman}\b', arabic, phase, flags=re.IGNORECASE)
    phase = re.sub(r'(?i)\bphase\s*', 'Phase ', phase)
    phase = re.sub(r'(\d[a-z]?)\s*/\s*(\d[a-z]?)', r'\1/Phase \2', phase)
    phase = re.sub(r'\s+And\s+', '/', phase)
    return phase

# Step 2: Create a new table with the desired columns
# Step 2: Create a new table with the desired columns, overwriting if it exists
conn.execute("DROP TABLE IF EXISTS final_table")
conn.execute("""
CREATE TABLE final_table AS
SELECT 
    i."Abstract Number",
    i."Extracted Indication" AS Indication,
    gt.potential_genes AS "Genes (incl. biomarkers)",
    gt.gene_target AS Targets,
    SUBSTRING(i."Abstract Text", 1, POSITION('Background' IN i."Abstract Text")) AS "Abstract Title",
    CASE 
        WHEN LOWER(p."Extracted Phase") = 'preclinical' THEN 'Preclinical (' || p."Preclinical model" || ')'
        ELSE p."Extracted Phase"
    END AS Phase,
    i."Abstract Text" AS "Full Abstract Text"
FROM indication i
LEFT JOIN gene_target gt ON i."Abstract Number" = gt."Abstract Number"
LEFT JOIN phase p ON i."Abstract Number" = p."Abstract Number"
WHERE LENGTH(gt.gene_target) > 2
""")

# Step 3: Fetch the data from the final_table
result = conn.execute("SELECT * FROM final_table").fetchdf()

# Step 4: Apply cleaning functions
result['Abstract Title'] = result['Full Abstract Text'].apply(extract_abstract_title)
result['Genes (incl. biomarkers)'] = result['Genes (incl. biomarkers)'].apply(handle_her2_erbb2)
result['Phase'] = result['Phase'].apply(standardize_phase)

# Step 5: Remove unnecessary columns
result = result.drop(columns=['Full Abstract Text'])

# Step 6: Write the result to a CSV file
result.to_csv('final_abstracts.csv', index=False)

# Step 7: Close the connection
conn.close()

print("Final table created, cleaned, and saved to 'final_abstracts.csv'")
