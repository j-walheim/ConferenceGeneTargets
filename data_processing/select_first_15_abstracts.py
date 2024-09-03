# %%
import pandas as pd

# Load the csv file
df = pd.read_csv('data/abstracts.csv')

# Select first 15 rows and save
df = df.head(15)

# Save to csv
df.to_csv('data/abstracts_15.csv', index=False)
# %%
