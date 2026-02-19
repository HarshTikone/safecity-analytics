"""
data_cleaning.py
Data Cleaning Module for LA Crime Data Analysis
EAS 587 - Phase 1 Project
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath):
    """Load raw crime data from CSV file."""
    print("Loading raw data...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns")
    return df

def clean_date_columns(df):
    """
    CLEANING OPERATION 1: Convert date columns to datetime format
    - Converts 'Date Rptd' and 'DATE OCC' to datetime objects
    - Extracts year, month, day, and day of week for analysis
    """
    print("\n[1] Cleaning date columns...")
    df['Date Rptd'] = pd.to_datetime(df['Date Rptd'])
    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])

    # Extract temporal features
    df['Year'] = df['DATE OCC'].dt.year
    df['Month'] = df['DATE OCC'].dt.month
    df['Day'] = df['DATE OCC'].dt.day
    df['DayOfWeek'] = df['DATE OCC'].dt.day_name()
    df['Hour'] = df['TIME OCC'] // 100

    print("  - Converted date columns to datetime")
    print("  - Extracted Year, Month, Day, DayOfWeek, Hour")
    return df

def clean_time_column(df):
    """
    CLEANING OPERATION 2: Handle invalid time values
    - Some TIME OCC values are > 2400 (invalid)
    - Filter and correct invalid time entries
    """
    print("\n[2] Cleaning time column...")
    invalid_times = df[df['TIME OCC'] > 2400].shape[0]
    print(f"  - Found {invalid_times} invalid time entries")

    # Cap at 2400 and convert to valid format
    df.loc[df['TIME OCC'] > 2400, 'TIME OCC'] = 2400
    df['Hour'] = df['TIME OCC'] // 100
    df.loc[df['Hour'] > 23, 'Hour'] = 23

    print("  - Corrected invalid time values")
    return df

def handle_missing_victim_info(df):
    """
    CLEANING OPERATION 3: Handle missing victim demographic data
    - Victim age 0 often indicates unknown (not actual age)
    - Replace 0 age with NaN for accurate statistics
    - Create 'Unknown' category for missing sex/descent
    """
    print("\n[3] Handling missing victim information...")

    # Age 0 is likely unknown, not actual age
    zero_age_count = (df['Vict Age'] == 0).sum()
    df.loc[df['Vict Age'] == 0, 'Vict Age'] = np.nan
    print(f"  - Replaced {zero_age_count} zero ages with NaN")

    # Fill missing sex/descent with 'Unknown'
    df['Vict Sex'] = df['Vict Sex'].fillna('Unknown')
    df['Vict Descent'] = df['Vict Descent'].fillna('Unknown')
    print("  - Filled missing sex/descent with 'Unknown'")

    return df

def standardize_categorical_values(df):
    """
    CLEANING OPERATION 4: Standardize categorical values
    - Standardize victim sex codes
    - Map descent codes to full descriptions
    """
    print("\n[4] Standardizing categorical values...")

    # Standardize sex codes
    sex_mapping = {
        'M': 'Male',
        'F': 'Female',
        'X': 'Unknown',
        'H': 'Unknown',
        'Unknown': 'Unknown'
    }
    df['Vict Sex Clean'] = df['Vict Sex'].map(sex_mapping)

    # Map descent codes to descriptions
    descent_mapping = {
        'W': 'White',
        'B': 'Black',
        'H': 'Hispanic',
        'A': 'Asian',
        'O': 'Other',
        'C': 'Chinese',
        'K': 'Korean',
        'J': 'Japanese',
        'F': 'Filipino',
        'V': 'Vietnamese',
        'I': 'American Indian',
        'Z': 'Asian Indian',
        'P': 'Pacific Islander',
        'U': 'Hawaiian',
        'D': 'Cambodian',
        'L': 'Laotian',
        'S': 'Samoan',
        'G': 'Guamanian',
        'X': 'Unknown',
        'Unknown': 'Unknown'
    }
    df['Vict Descent Clean'] = df['Vict Descent'].map(descent_mapping)

    print("  - Standardized victim sex codes")
    print("  - Mapped descent codes to descriptions")
    return df

def remove_unused_crime_code_columns(df):
    """
    CLEANING OPERATION 5: Remove columns with excessive missing data
    - Crm Cd 2, 3, 4 are 98-100% missing
    - These provide no analytical value
    """
    print("\n[5] Removing columns with excessive missing data...")
    cols_to_drop = ['Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"  - Dropped columns: {cols_to_drop}")
    return df

def categorize_crime_types(df):
    """
    CLEANING OPERATION 6: Create crime category groupings
    - Group similar crime types into broader categories
    - Makes analysis more manageable and interpretable
    """
    print("\n[6] Categorizing crime types...")

    def get_crime_category(crime_desc):
        crime_desc = str(crime_desc).upper()
        if 'VEHICLE' in crime_desc or 'STOLEN' in crime_desc:
            return 'Vehicle Crime'
        elif 'THEFT' in crime_desc or 'SHOPLIFTING' in crime_desc or 'PICKPOCKET' in crime_desc:
            return 'Theft'
        elif 'BURGLARY' in crime_desc:
            return 'Burglary'
        elif 'VANDALISM' in crime_desc:
            return 'Vandalism'
        elif 'ASSAULT' in crime_desc or 'BATTERY' in crime_desc:
            return 'Assault/Battery'
        elif 'ROBBERY' in crime_desc:
            return 'Robbery'
        elif 'IDENTITY' in crime_desc:
            return 'Identity Theft'
        elif 'TRESPASS' in crime_desc:
            return 'Trespassing'
        elif 'DRUG' in crime_desc or 'NARCOTIC' in crime_desc:
            return 'Drug Offense'
        elif 'FRAUD' in crime_desc or 'EMBEZZLE' in crime_desc or 'BUNCO' in crime_desc:
            return 'Fraud'
        elif 'SEX' in crime_desc or 'RAPE' in crime_desc or 'LEWD' in crime_desc:
            return 'Sex Offense'
        elif 'HOMICIDE' in crime_desc or 'MURDER' in crime_desc:
            return 'Homicide'
        else:
            return 'Other'

    df['Crime Category'] = df['Crm Cd Desc'].apply(get_crime_category)
    print("  - Created 'Crime Category' column with 12 categories")
    print(f"  - Categories: {df['Crime Category'].unique().tolist()}")
    return df

def categorize_premise_types(df):
    """
    CLEANING OPERATION 7: Categorize premise types
    - Group premise descriptions into broader location categories
    """
    print("\n[7] Categorizing premise types...")

    def get_premise_category(premise):
        if pd.isna(premise):
            return 'Unknown'
        premise = str(premise).upper()
        if 'STREET' in premise or 'SIDEWALK' in premise or 'ALLEY' in premise:
            return 'Public Street'
        elif 'PARKING' in premise or 'GARAGE' in premise or 'DRIVEWAY' in premise:
            return 'Parking Area'
        elif 'STORE' in premise or 'MARKET' in premise or 'BUSINESS' in premise or 'SHOP' in premise:
            return 'Commercial'
        elif 'DWELLING' in premise or 'HOUSE' in premise or 'APARTMENT' in premise or 'RESIDENCE' in premise:
            return 'Residential'
        elif 'VEHICLE' in premise:
            return 'Vehicle'
        elif 'MTA' in premise or 'TRANSPORT' in premise or 'BUS' in premise or 'TRAIN' in premise:
            return 'Transit'
        elif 'PARK' in premise or 'PLAYGROUND' in premise:
            return 'Park/Recreation'
        elif 'SCHOOL' in premise or 'COLLEGE' in premise or 'UNIVERSITY' in premise:
            return 'Educational'
        else:
            return 'Other'

    df['Premise Category'] = df['Premis Desc'].apply(get_premise_category)
    print("  - Created 'Premise Category' column")
    return df

def create_age_groups(df):
    """
    CLEANING OPERATION 8: Create age group categories
    - Bin victim ages into meaningful groups for analysis
    """
    print("\n[8] Creating age groups...")

    bins = [0, 18, 25, 35, 45, 55, 65, 100]
    labels = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']

    df['Age Group'] = pd.cut(df['Vict Age'], bins=bins, labels=labels, right=False)
    print("  - Created 'Age Group' column with 7 categories")
    return df

def handle_coordinate_outliers(df):
    """
    CLEANING OPERATION 9: Handle geographic coordinate outliers
    - LA coordinates should be approximately 33-35 N, -117 to -119 W
    - Remove or flag entries outside reasonable bounds
    """
    print("\n[9] Handling coordinate outliers...")

    # LA area bounds
    lat_min, lat_max = 33.5, 34.5
    lon_min, lon_max = -118.8, -117.8

    outlier_mask = (
        (df['LAT'] < lat_min) | (df['LAT'] > lat_max) |
        (df['LON'] < lon_min) | (df['LON'] > lon_max)
    )
    outlier_count = outlier_mask.sum()

    # Flag outliers instead of removing
    df['Valid Coordinates'] = ~outlier_mask
    print(f"  - Flagged {outlier_count} records with outlier coordinates")
    return df

def create_reporting_delay_feature(df):
    """
    CLEANING OPERATION 10: Calculate reporting delay
    - Days between crime occurrence and reporting
    - Important feature for understanding reporting patterns
    """
    print("\n[10] Creating reporting delay feature...")

    df['Reporting Delay (Days)'] = (df['Date Rptd'] - df['DATE OCC']).dt.days

    # Flag negative delays (data quality issue)
    negative_delays = (df['Reporting Delay (Days)'] < 0).sum()
    print(f"  - Found {negative_delays} records with negative reporting delay")

    # Set negative delays to 0 (same-day reporting)
    df.loc[df['Reporting Delay (Days)'] < 0, 'Reporting Delay (Days)'] = 0

    print("  - Created 'Reporting Delay (Days)' column")
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV."""
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    print(f"Final dataset: {len(df):,} records, {len(df.columns)} columns")

def main():
    """Main cleaning pipeline."""
    # Load data
    df = load_data('data/raw/crime_data_2024_to_present.csv')

    # Apply cleaning operations
    df = clean_date_columns(df)
    df = clean_time_column(df)
    df = handle_missing_victim_info(df)
    df = standardize_categorical_values(df)
    df = remove_unused_crime_code_columns(df)
    df = categorize_crime_types(df)
    df = categorize_premise_types(df)
    df = create_age_groups(df)
    df = handle_coordinate_outliers(df)
    df = create_reporting_delay_feature(df)

    # Save cleaned data
    save_cleaned_data(df, 'data/processed/crime_data_cleaned.csv')

    return df

if __name__ == "__main__":
    main()
