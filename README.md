# SafeCity Analytics: LA Crime Data (Phase 1)

## Course + Assignment Header
- **Subject Code:** EAS 587  
- **Course:** Data-Intensive Computing (Spring 2026)  
- **Assignment No.:** Assignment 1 (Project Phase 1)  
- **Project Title:** SafeCity Analytics: LA Crime Data Analysis  
- **Instructor:** Dr. Justice Del Vacio 
- **Team Members:**  
  - Harsh Mahesh Tikone  
  - Dev Desai  
  - Shwetangi  

---

## Report & Deliverables

| Deliverable | Link / File |
|---|---|
| **Phase 1 Report (Google Doc)** | [View Report](https://docs.google.com/document/d/1IfdWrU0ViWzt-P31DnCWtzFBuykYbGcUl0mOn432MX8/edit?usp=sharing) |
| **Workshop Slides** | `LA_Crime_Data_Analysis.pptx` |

---

## 1) Project Overview
This project implements the Phase 1 pipeline for structured crime data analysis using Python:
- Data ingestion check
- Reproducible data cleaning/processing
- Exploratory Data Analysis (EDA) with tables + visualizations

Primary dataset used:
- **Source:** [Crime Data from 2020 to Present (data.gov)](https://catalog.data.gov/dataset/crime-data-from-2020-to-present)
- **File in repo:** `data/raw/crime_data_2024_to_present.csv`
- **Scale:** ~62K rows (meets 50,000+ row requirement)

The code is designed to run from a **fresh local Python environment** and generate all processed outputs in `data/processed/`.

---

## 2) Repository Structure
```text
DIC_Assignment_safecity-analytics/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # Original data files
│   │   └── crime_data_2024_to_present.csv
│   └── processed/            # Cleaned data files
│       └── crime_data_cleaned.csv
├── src/
│   ├── data_cleaning.py      # Data cleaning pipeline (10 operations)
│   └── eda.py                # EDA analysis (10 operations)
└── figures/                  # Generated visualizations
    ├── temporal_patterns.png
    ├── geographic_distribution.png
    ├── victim_demographics.png
    ├── crime_type_analysis.png
    ├── reporting_patterns.png
    ├── cross_tabulation.png
    ├── correlation_matrix.png
    ├── outlier_detection.png
    └── weapon_analysis.png
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd project-repo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
### Running the Analysis

1. **Data Cleaning:**
```bash
cd src
python data_cleaning.py
```

2. **Exploratory Data Analysis:**
```bash
python eda.py
```

## Data Cleaning Operations (10)

1. **Date Column Conversion:** Converted `Date Rptd` and `DATE OCC` to datetime format; extracted Year, Month, Day, DayOfWeek, Hour
2. **Time Validation:** Fixed invalid time values (>2400)
3. **Missing Victim Info:** Replaced zero ages with NaN; filled missing sex/descent with 'Unknown'
4. **Categorical Standardization:** Standardized sex codes (M→Male, F→Female, X→Unknown); mapped descent codes to full descriptions
5. **Column Removal:** Removed `Crm Cd 2`, `Crm Cd 3`, `Crm Cd 4` (98-100% missing)
6. **Crime Categorization:** Grouped 140+ crime types into 12 categories (Vehicle Crime, Theft, Burglary, etc.)
7. **Premise Categorization:** Grouped premise types into 9 categories (Public Street, Parking Area, Commercial, etc.)
8. **Age Grouping:** Created 7 age groups (0-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+)
9. **Coordinate Validation:** Flagged coordinates outside LA bounds (none found)
10. **Reporting Delay:** Calculated days between crime occurrence and report

## EDA Operations (10) - Following John Tukey's Principles

1. **Summary Statistics:** Generated descriptive statistics for numeric variables
2. **Temporal Patterns:** Analyzed crime by hour, day of week, and month
3. **Geographic Distribution:** Mapped crimes by LAPD area and coordinates
4. **Victim Demographics:** Analyzed age, sex, and descent distributions
5. **Crime Type Analysis:** Examined crime categories and premise types
6. **Reporting Patterns:** Analyzed reporting delays and case statuses
7. **Cross-tabulation:** Crime categories by victim sex and area
8. **Correlation Analysis:** Correlation matrix of numeric variables
9. **Outlier Detection:** Box plots for age, reporting delay, and hour
10. **Weapon Analysis:** Weapon usage patterns and types

## Key Findings

### Temporal Patterns
- **Peak crime hour:** 6:00 PM (3,911 crimes)
- **Highest crime day:** Friday (9,550 crimes)
- **Highest crime month:** May (9,388 crimes)

### Geographic Distribution
- **Highest crime area:** Central LA (6,024 crimes)
- **Lowest crime area:** Foothill (1,774 crimes)

### Crime Types
- **Top category:** Vehicle Crime (28,700 crimes, 46.2%)
- **Top premise:** Public Street (23,518 crimes, 37.9%)

### Victim Demographics
- **Median age:** 35 years
- **Sex distribution:** 37.6% Male, 29.3% Female, 33.1% Unknown
- **Top descent:** Hispanic (10,399), Black (4,089), Other (3,666)

### Reporting
- **Median reporting delay:** 1 day
- **Case status:** 94.5% under investigation

## Surprise Findings

1. **High proportion of unknown victim data:** 48.5% of records have unknown victim demographics (age=0, sex=X), suggesting many crimes are reported without victim information (e.g., property crimes)

2. **Low weapon usage:** Only 5.9% of crimes involve weapons, with "strong-arm" (physical force) being the most common

3. **Quick reporting:** 75% of crimes are reported within 3 days, indicating relatively prompt reporting

## Dead Ends

1. **Attempted crime analysis:** Initially tried to analyze attempted vs. completed crimes using crime codes, but the distinction was inconsistent across crime types and difficult to categorize reliably.

2. **Seasonal trend analysis:** Attempted to analyze seasonal patterns, but the dataset only covers ~1 year (2024), making seasonal comparisons impossible.

3. **Victim-offender relationship analysis:** The dataset doesn't include offender information or relationship data, preventing analysis of crime dynamics.

## Design Decisions

1. **Age handling:** Replaced age=0 with NaN rather than imputing, as 0 likely represents "unknown" rather than actual age
2. **Crime categorization:** Created broad categories to simplify analysis of 140+ crime types while maintaining interpretability
3. **Coordinate outliers:** Flagged rather than removed outliers to preserve data integrity
4. **Visualization approach:** Used multiple chart types (bar, line, pie, scatter) to reveal different aspects of the data

## Dependencies
See `requirements.txt` for full list:
- pandas
- numpy
- matplotlib
- seaborn

## References
1. Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
2. Los Angeles Police Department. Crime Data from 2020 to Present. https://data.lacity.org/

## License
This project is for educational purposes (EAS 587).
