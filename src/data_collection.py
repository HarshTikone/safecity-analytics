"""
Data Collection Script
"""

import pandas as pd

# Load data
df = pd.read_csv('data/raw/crime_data.csv')

print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print(df.head())
