# EAS 587 Phase 1: Crime Data Analysis

# Team Members 
1. Harsh Mahesh Tikone
2. Dev Desai
3. Shwetangi Singh

## Project Overview
Analysis of crime data from 2020 to present.

**Data Source:** https://catalog.data.gov/dataset/crime-data-from-2020-to-present

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download data from the link above and place in `data/raw/`
4. Run scripts:
   ```bash
   python src/data_collection.py
   python src/data_cleaning.py
   python src/eda.py
   ```

## Project Structure
```
project-repo/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
└── src/
    ├── data_collection.py
    ├── data_cleaning.py
    └── eda.py
```
