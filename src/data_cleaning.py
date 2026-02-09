"""
Data Cleaning Script
"""

import pandas as pd

# Load raw data
df = pd.read_csv('data/raw/crime_data.csv')

# TODO: Add your 10+ cleaning operations here
# Example:
# 1. Remove duplicates
df = df.drop_duplicates()

# 2. Handle missing values
# df = df.dropna()

# Save cleaned data
df.to_csv('data/processed/crime_data_cleaned.csv', index=False)
print(f"Cleaned data saved: {len(df)} rows")
