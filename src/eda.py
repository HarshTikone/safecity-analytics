"""
Exploratory Data Analysis Script
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv('data/processed/crime_data_cleaned.csv')

# TODO: Add your 10+ EDA operations here
# Example:
print(df.describe())
print(df.info())

# Create some visualizations
# plt.figure(figsize=(10, 6))
# df['column_name'].value_counts().plot(kind='bar')
# plt.savefig('output.png')
