# data_cleaning.py

import pandas as pd

# Load data (example: from a URL or local file)
url = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
df = pd.read_csv(url)

# Cleaning example
df.dropna(inplace=True)
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

print("Data cleaned and saved.")
