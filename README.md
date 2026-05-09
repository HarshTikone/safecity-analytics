# SafeCity Analytics: LA Crime Data Analysis

## Course + Assignment Header
- **Subject Code:** EAS 587  
- **Course:** Data-Intensive Computing (Spring 2026)  
- **Assignment No.:** Project Phase 1, Phase 2, Phase 3 & Phase 4
- **Project Title:** SafeCity Analytics: LA Crime Data Analysis  
- **Instructor:** Dr. Justice Del Vacio  
- **Team Members:**  
  - Harsh Mahesh Tikone  
  - Dev Desai  
  - Shwetangi Singh 

---

## Report & Deliverables

| Deliverable | Link / File |
|---|---|
| **Phase 1 Report (Google Doc)** | [View Report](https://docs.google.com/document/d/1oYahBmjBAiVArPI48sZtJC_sIqrrbvFXyusX9ByZsmY/edit?usp=sharing) |
| **Phase 1 Workshop Slides** | `LA_Crime_Data_Analysis.pptx` |
| **Phase 2 Report (Google Doc)** | [View Report](https://docs.google.com/document/d/10sJOqEEXB30xsa94dIkb-TCN8guEs0muV1l9IHjfs6g/edit?tab=t.0) |
| **Phase 2 Workshop Slides** | `safecity_workshop_slides_phase_2.pptx` |
| **Phase 3 Report (Google Doc)** | [View Report](https://docs.google.com/document/d/1DYWHYWxX1tqeKoY19AvVmRuRjts-9Jhkl3FwlD8IBaY/edit?tab=t.0) |
| **Phase 3 Workshop Slides** | `LA_Crime_Data_Analysis.pptx` |
| **Phase 4 Report (PDF)** | `Phase4report.pdf` |
| **Phase 4 Presentation Slides (PDF)** | `presentation_slides.pdf` |

---

## How to Use This Project (Quick Start)

Run all commands from the project root (`safecity-analytics/`).

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run Phase 1 (cleaning + EDA)

```bash
python3 src/data_cleaning.py
python3 src/eda.py
```

- Cleaned dataset: `data/processed/crime_data_cleaned.csv`
- Cleaning audit log: `data/cleaning_audit.json`
- EDA plots: `figures/`

### 3) Run Phase 2 (model training + comparison)

```bash
python3 src/models/train_knn.py
python3 src/models/train_decision_tree.py
python3 src/models/train_kmeans.py
python3 src/models/train_naive_bayes.py
python3 src/models/train_random_forest.py
python3 src/models/train_logistic_regression.py
python3 src/models/compare_algorithms.py
```

- Trained models: `models/`
- Metrics/figures: `outputs/`

### 4) Start MCP server

```bash
python3 src/mcp/server.py
```

### 5) Use MCP in Claude Code (optional)

```bash
claude mcp add -s project safecity-crime-predictor -- python3 src/mcp/server.py
claude mcp get safecity-crime-predictor
```

### 6) Run Phase 3 (Databricks / Spark)

Import notebooks from `notebooks/databricks/` into your Databricks workspace and run in order (see Phase 3 section below).

---

## Repository Structure

```text
safecity-analytics/
├── README.md
├── requirements.txt
├── .gitattributes
├── .gitignore
├── .mcp.json
├── LA_Crime_Data_Analysis.pptx
├── safecity_workshop_slides_phase_2.pptx
├── Phase4report.pdf                          ← Phase 4: final report
├── presentation_slides.pdf                   ← Phase 4: final presentation
├── data/                                     ← Phase 1: raw and processed data
│   ├── raw/
│   │   └── crime_data_2024_to_present.csv
│   ├── processed/
│   │   └── crime_data_cleaned.csv
│   └── cleaning_audit.json
├── figures/                                  ← Phase 1: EDA visualizations
├── models/                                   ← Phase 2: serialized trained models
│   ├── knn_model.pkl
│   ├── decision_tree_model.pkl
│   ├── kmeans_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   └── logistic_regression_model.pkl
├── outputs/                                  ← Phase 2: plots and metrics CSVs
│   ├── comparison/
│   ├── decision_tree/
│   └── kmeans/
├── src/                                      ← Phase 1 & 2: source code
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── models/
│   │   ├── preprocess.py
│   │   ├── train_knn.py
│   │   ├── train_decision_tree.py
│   │   ├── train_kmeans.py
│   │   ├── train_naive_bayes.py
│   │   ├── train_random_forest.py
│   │   ├── train_logistic_regression.py
│   │   └── compare_algorithms.py
│   └── mcp/
│       ├── server.py
│       └── README.md
├── notebooks/
│   └── databricks/                           ← Phase 3: Databricks notebooks
│       ├── 01_lapd_bronze.ipynb
│       ├── 02_lapd_silver.ipynb
│       ├── 03_lapd_gold.ipynb
│       ├── 04_lapd_mllib.ipynb
│       ├── 05_nibrs_bronze.ipynb
│       ├── 06_nibrs_silver_combined.ipynb
│       ├── 07_nibrs_insights.ipynb
│       └── 08_nibrs_mllib.ipynb
└── part2_phase3_databrick_model/             ← Phase 3: saved MLlib pipeline model
    ├── metadata/
    └── stages/
        ├── 0_VectorAssembler_.../
        ├── 1_StandardScaler_.../
        └── 2_LogisticRegression_.../
```

---

# PHASE 1: Data Collection, Cleaning & EDA

---

## Phase 1 Overview

Phase 1 covers the data ingestion, cleaning, and exploratory analysis steps of the pipeline. The goal was to get the raw LAPD crime data into a clean, analysis-ready state and surface enough patterns to guide our modeling choices in Phase 2.

We used a single primary dataset:
- **Source:** [Crime Data from 2020 to Present (data.gov)](https://catalog.data.gov/dataset/crime-data-from-2020-to-present)
- **File in repo:** `data/raw/crime_data_2024_to_present.csv`
- **Scale:** ~62K rows (meets the 50,000+ row requirement)

---

## Phase 1 Setup Instructions

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
git clone <repository-url>
cd safecity-analytics
pip install -r requirements.txt
```

### Running Phase 1

**1. Data Cleaning:**
```bash
python3 src/data_cleaning.py
```

**2. Exploratory Data Analysis:**
```bash
python3 src/eda.py
```

---

## Data Cleaning Operations (10)

1. **Date Column Conversion:** Converted `Date Rptd` and `DATE OCC` to datetime format; extracted Year, Month, Day, DayOfWeek, Hour
2. **Time Validation:** Fixed invalid time values (>2400)
3. **Missing Victim Info:** Replaced zero ages with NaN; filled missing sex/descent with 'Unknown'
4. **Categorical Standardization:** Standardized sex codes (M→Male, F→Female, X→Unknown); mapped descent codes to full descriptions
5. **Column Removal:** Removed `Crm Cd 2`, `Crm Cd 3`, `Crm Cd 4` (98–100% missing)
6. **Crime Categorization:** Grouped 140+ crime types into 12 categories (Vehicle Crime, Theft, Burglary, etc.)
7. **Premise Categorization:** Grouped premise types into 9 categories (Public Street, Parking Area, Commercial, etc.)
8. **Age Grouping:** Created 7 age groups (0–17, 18–24, 25–34, 35–44, 45–54, 55–64, 65+)
9. **Coordinate Validation:** Flagged coordinates outside LA bounds (none found)
10. **Reporting Delay:** Calculated days between crime occurrence and report

---

## EDA Operations (10) — Following John Tukey's Principles

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

---

## Phase 1 Key Findings

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

---

## Phase 1 Surprise Findings

A few things caught us off guard during EDA.

Nearly half the records (48.5%) had unknown victim demographics — age listed as 0, sex as X. This makes sense once you realize a lot of these are property crimes where no victim is directly identified at the scene, but it was still a bigger gap than we expected and shaped how we handled those fields downstream.

Weapon involvement was also much lower than anticipated. Only 5.9% of crimes involved a weapon, and even then "strong-arm" (physical force, no weapon) was the most common type. This turned out to be useful — it meant predicting weapon involvement was a meaningful but tractable binary classification problem for Phase 2.

On the positive side, reporting was faster than expected. Three-quarters of crimes were reported within 3 days, which suggests the data is relatively fresh and not heavily skewed by delayed reports.

---

## Phase 1 Dead Ends

Not everything we tried worked out.

We initially tried to separate attempted crimes from completed ones using the crime codes, but the distinction was inconsistently applied across crime types and wasn't reliable enough to use as a feature or label.

We also looked at seasonal trends but quickly realized the dataset only covers about one year (2024), so there wasn't enough history for meaningful seasonal analysis.

Finally, we wanted to look at victim-offender relationships but the dataset simply doesn't include offender data, so that line of analysis wasn't possible.

---

## Phase 1 Design Decisions

A few choices we made deliberately and why:

We replaced age=0 with NaN rather than imputing a value — 0 almost certainly means "unknown" in this dataset, not an actual age, so filling it with a mean or median would have introduced noise.

We grouped 140+ crime types into broader categories to keep the analysis readable and the models tractable. Fine-grained crime codes would have created a very sparse label space.

For coordinate outliers, we flagged them rather than dropping them. Removing data points without a clear reason felt like the wrong call, and none of the flagged points turned out to be outside LA bounds anyway.

For visualizations, we mixed chart types on purpose — bar charts for counts, line charts for trends, scatter plots for geography — because different structures in the data show up better in different chart types.

---

# PHASE 2: Machine Learning & Statistical Analysis

---

## Phase 2 Overview

Phase 2 takes the cleaned dataset from Phase 1 and applies six ML algorithms to it, each targeting a different question about crime in LA. We also deployed one model as an MCP server and did a head-to-head comparison of the algorithms.

- **Input:** `data/processed/crime_data_cleaned.csv` (Phase 1 output)
- **Scale:** ~62K rows, 46 features

---

## Phase 2 Setup Instructions

Dependencies are the same as Phase 1, plus scikit-learn and mcp:

```bash
pip install -r requirements.txt
```

### Running Phase 2

Run all scripts from the project root in this order:

```bash
python3 src/models/train_knn.py
python3 src/models/train_decision_tree.py
python3 src/models/train_kmeans.py
python3 src/models/train_naive_bayes.py
python3 src/models/train_random_forest.py
python3 src/models/train_logistic_regression.py
python3 src/models/compare_algorithms.py
```

All plots are saved to `outputs/<algorithm>/` and all serialized models to `models/`.

### Running the MCP Server

```bash
# Step 1: Train Random Forest first (if not already done)
python3 src/models/train_random_forest.py

# Step 2: Start the MCP server
python3 src/mcp/server.py
```

See `src/mcp/README.md` for Claude Code integration instructions.

---

## ML Algorithms (6)

### In-Class Algorithms

1. **k-Nearest Neighbours (kNN)** — Predicts crime **severity** (High/Medium/Low) from time, location, and context features. k tuned via 5-fold cross-validation (k=3 to 15). Features scaled with StandardScaler (distance-based model requires scaling).

2. **Decision Tree** — Predicts **crime category** (13 classes) and produces human-readable decision rules. Hyperparameters tuned with GridSearchCV across `max_depth` and `min_samples_split`. Outputs tree structure visualization and feature importances.

3. **k-Means Clustering** — Unsupervised discovery of geographic **crime hotspots** across Los Angeles. Optimal k selected by combining elbow method (inertia) and silhouette score. Outputs a lat/lon cluster map with crime category breakdown per cluster.

4. **Naive Bayes** — Fast probabilistic prediction of **crime category** from categorical features. Compares GaussianNB vs. ComplementNB via cross-validation; ComplementNB selected for better handling of class imbalance across the 13 categories.

### Outside-Class Algorithms

5. **Random Forest** *(Breiman, 2001)* — Ensemble of Decision Trees predicting **crime category**. Reduces overfitting vs. a single DT; handles class imbalance via `class_weight='balanced'`. Tuned with RandomizedSearchCV. **Also deployed as the MCP server model.**

6. **Logistic Regression** *(Hosmer & Lemeshow, 2000)* — Binary classification predicting **weapon involvement** (True/False). Outputs calibrated probabilities for risk scoring. Regularisation strength C tuned via cross-validation. Evaluated with ROC-AUC and Precision-Recall curves.

---

## Algorithms Summary Table

| # | Algorithm | Type | Target Variable | Source |
|---|-----------|------|-----------------|--------|
| 1 | k-Nearest Neighbours | Classification | Severity (High/Med/Low) | In-class |
| 2 | Decision Tree | Classification | Crime Category (13 classes) | In-class |
| 3 | k-Means | Clustering | Geographic Hotspots | In-class |
| 4 | Naive Bayes | Classification | Crime Category (13 classes) | In-class |
| 5 | Random Forest | Classification | Crime Category (13 classes) | Outside class |
| 6 | Logistic Regression | Classification | Weapon Involvement (binary) | Outside class |

---

## Features Used

All supervised models share a common feature set built in `preprocess.py`:

| Feature | Type | Description |
|---------|------|-------------|
| `AREA` | Numeric | LAPD area code (1–21) |
| `Hour` | Numeric | Hour of day the crime occurred |
| `Month` | Numeric | Month of occurrence |
| `IsWeekend` | Binary | 1 if Saturday or Sunday |
| `Has Weapon` | Binary | 1 if a weapon was used |
| `Premise Category` | Encoded | Commercial / Residential / Public Street / etc. |
| `TimeBucket` | Encoded | Morning / Afternoon / Evening / Night |
| `Severity` | Encoded | High / Medium / Low (used when predicting category) |
| `Part 1-2` | Numeric | LAPD crime seriousness classification |
| `Reporting Delay (Days)` | Numeric | Days between crime and report |

---

## MCP Deployment

The trained **Random Forest** model is deployed as an MCP (Model Context Protocol) server, making it callable from Claude Desktop or any MCP-compatible AI assistant.

- **Exposed tools:** `predict_crime_category`, `list_crime_categories`, `server_health`
- **Input:** Area, hour, month, weekend flag, weapon flag, premise type, time bucket, severity, part classification, reporting delay
- **Output:** Predicted crime category + top-3 probability breakdown
- **Fallback:** If the model file is missing, the server retrains automatically from the cleaned CSV

Full setup instructions: [`src/mcp/README.md`](src/mcp/README.md)

---

## Phase 2 Key Results

### Classification Performance

| Algorithm | Target | Test Accuracy | Weighted F1 | 3-Fold CV Acc |
|-----------|--------|--------------|-------------|---------------|
| kNN (best k=3) | Severity | 0.9985 | 1.00 | — |
| kNN (k=7, comparison) | Crime Category | 0.7946 | 0.7771 | 0.785 |
| Decision Tree | Crime Category | 0.8114 | 0.7997 | 0.8014 |
| Naive Bayes | Crime Category | 0.5104 | 0.3961 | 0.5058 |
| Random Forest | Crime Category | 0.8150 | 0.8063 | 0.8137 |
| Logistic Regression | Weapon Involved | 0.8674 | 0.89 | — |

### Clustering Performance

| Algorithm | Best k | Silhouette Score |
|-----------|--------|--------------------|
| k-Means | 2 | 0.5456 |

---

## Phase 2 Dead Ends

Two approaches didn't work out and are worth documenting.

We tried SVM for crime category classification, but even `LinearSVC` took over 30 minutes on the full 62K-row dataset. Downsampling was an option but would have underrepresented rare crime types, which felt like the wrong trade-off. We switched to Random Forest, which trains faster and handles class imbalance more cleanly.

We also tried DBSCAN for geographic clustering as an alternative to k-Means. It ended up labeling over 60% of points as noise because crime in LA is spread fairly uniformly across the city — there are no tight, isolated clusters for DBSCAN to latch onto. The results weren't geographically meaningful, so we went back to k-Means with silhouette-guided k selection.

---

## Phase 2 Design Decisions

A few deliberate choices worth explaining:

We built a shared `preprocess.py` module that all six models pull from. This keeps feature encoding consistent across scripts and avoids the kind of subtle data leakage that can happen when each script does its own encoding independently.

We chose Random Forest for the MCP deployment over Decision Tree because it generalizes better and handles class imbalance — important for a model that will be called at inference time with real inputs.

Logistic Regression was applied to weapon involvement (binary) rather than crime category. This added variety to the algorithm set and produced something genuinely useful — a calibrated probability score for weapon risk rather than just a category label.

We chose ComplementNB over GaussianNB for the Naive Bayes model because it's designed for imbalanced multi-class problems, which fits our data well — Vehicle Crime makes up 46% of records while some categories are under 1%.

For k-Means, we used both the elbow method and silhouette score together. The elbow method alone often gives ambiguous results; the silhouette score adds a measure of actual cluster separation, which helped us pick a k that was both efficient and meaningful.

---

## PHASE 3: Distributed Computing with Databricks & Spark
 
### Overview
 
Phase 3 moves the pipeline to Databricks / Apache Spark and integrates a second source — FBI's National Incident-Based Reporting System (NIBRS) — for cross-dataset analysis.
 
The pipeline follows a **Medallion architecture** across three Delta Lake layers:
- **Bronze** — raw ingestion, no transforms, with `_source` and `_ingested_at` audit columns
- **Silver** — cleaned, type-cast, and feature-engineered; all three source tables left-joined on `CaseNo` into `silver_lapd_crimes` (62,105 rows, partitioned by AREA across 21 divisions)
- **Gold** — six aggregated tables, each answering a specific analytical question
13 Delta tables total across all three layers.
 
### Running Phase 3
 
Import notebooks from `notebooks/databricks/` and run in order:
 
| Order | Notebook | What it does |
|-------|----------|--------------|
| 1 | `01_lapd_bronze.ipynb` | Ingest LAPD CSV → `bronze_mydata` |
| 2 | `02_lapd_silver.ipynb` | Clean + transform → `silver_mydata` |
| 3 | `03_lapd_gold.ipynb` | Aggregates → 6 gold Delta tables |
| 4 | `04_lapd_mllib.ipynb` | Decision Tree + Naive Bayes on primary data |
| 5 | `05_nibrs_bronze.ipynb` | Ingest NIBRS CSVs → bronze Delta tables |
| 6 | `06_nibrs_silver_combined.ipynb` | Join LAPD + NIBRS → `silver_lapd_crimes` |
| 7 | `07_nibrs_insights.ipynb` | 3 insights from combined data |
| 8 | `08_nibrs_mllib.ipynb` | Logistic Regression on weapon prediction |
 
### Phase 3 Key Results
 
| Algorithm | Target | Phase 2 (sklearn) | Phase 3 (MLlib) |
|-----------|--------|-------------------|-----------------|
| Decision Tree | Crime Category | Acc=0.8114 / F1=0.7997| Acc=0.5285 / F1=0.5044 |
| Naive Bayes | Crime Category | Acc=0.5104 / F1=0.3961 | Acc=0.3967 / F1=0.3058 |
| Logistic Regression | Weapon (binary) | ROC-AUC=0.8674 / Acc=0.8674 | ROC-AUC=0.9123 / Acc=0.9385 / F1=0.9205 |
 
The Decision Tree and Naive Bayes accuracy drops are explained by distributed shuffling in Spark vs. pandas, no cross-validation on CE (OOM constraint), and an 80/20 split vs. the 60/20/20 split in Phase 2. The Logistic Regression weapon predictor improved in Phase 3, reaching ROC-AUC of 0.9123 with NIBRS-augmented features.
 
### NIBRS Insights
 
**Insight 1 — Weapon crime is concentrated:**
Pacific division has a 20.37% weapon rate (913 of 4,481 crimes). The next highest is Southeast at 9.36%. NIBRS shows most offenses are crimes against property (137,367) but person-targeted crimes (85,820) carry most weapon involvement.
 
**Insight 2 — Both sources agree on when crime happens:**
LAPD and NIBRS hourly distributions, normalized to percentage share, are nearly identical — both dip at 5–6am and peak in the evening. Weekend average (crimes/day) is slightly higher than weekday.
 
**Insight 3 — Victim demographics and reporting delays:**
83.6% of NIBRS victims are Person-type. In LAPD, 48.5% of victim demographic records are unknown. Financial crimes go unreported the longest — embezzlement averages 31 days, identity theft 18 days. Violent crimes are reported within a day or two.
 
### Challenges & Dead Ends
 
**StringIndexer blocked on CE Shared clusters.** Databricks Community Edition's Shared cluster blocks StringIndexer via the Py4J whitelist. Fixed by using Spark SQL `dense_rank()` window functions instead — same integer indices, works on any cluster type.
 
**CrossValidator caused OOM.** The CE node crashed when caching multiple model copies. Fixed hyperparameters were used instead (`regParam=0.01` for LR, manual `maxDepth=15` for DT). Model objects were deleted from memory after evaluation to prevent subsequent cells from crashing.
 
**CaseNo join is partial.** LAPD `DR_NO` and NIBRS `CaseNo` use different formatting — only 45 of 62,105 rows (0.1%) matched directly. We analyzed the NIBRS tables independently and compared distributions rather than relying solely on joined rows.
 
### Saved MLlib Model
 
```python
from pyspark.ml import PipelineModel
model = PipelineModel.load("part2_phase3_databrick_model/")
```
 
---
# PHASE 4: Final Report & Presentation

---

## Phase 4 Overview

Phase 4 consolidates findings from all prior phases into a final written report and presentation delivered to the class. No new code is introduced; this phase synthesizes the full project narrative — data pipeline, modeling results, distributed computing, and key insights — into polished deliverables.

- **Final Report:** `Phase4report.pdf`
- **Presentation Slides:** `presentation_slides.pdf`

---

## Reproducibility

- All random seeds are set to `42` across all Phase 2 scripts
- `preprocess.py` provides a single shared feature pipeline used by all models
- Run scripts in the order listed under **Running Phase 2** above
- Verified on a fresh environment: confirmed by Harsh on 21st March

---

## Dependencies

See `requirements.txt` for full pinned versions:

```
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
scipy==1.13.0
scikit-learn==1.5.0
mcp>=1.9.0
```

---

## References

1. Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
2. Los Angeles Police Department. Crime Data from 2020 to Present. https://data.lacity.org/
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
4. Hosmer, D.W. & Lemeshow, S. (2000). *Applied Logistic Regression* (2nd ed.). Wiley. https://doi.org/10.1002/0471722146
5. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830. https://scikit-learn.org
6. Model Context Protocol Documentation. https://modelcontextprotocol.io/
7. VanderPlas, J. (2016). *Python Data Science Handbook*. O'Reilly. https://jakevdk.github.io/PythonDataScienceHandbook/
8. Apache Spark MLlib Documentation. https://spark.apache.org/docs/latest/ml-guide.html

---
