"""
data_cleaning.py
Production-Grade Data Cleaning Pipeline for LA Crime Data Analysis
EAS 587 - Phase 1 Project

Design principles:
- Every transformation is logged with before/after counts
- No silent data loss — all decisions are documented
- Functions are pure (input → output), no global state
- A single `run_pipeline()` call reproduces results end-to-end
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# LA bounding box (USGS verified)
LA_LAT = (33.5, 34.5)
LA_LON = (-118.95, -117.65)

# Victim age: 0 = unknown in LAPD encoding; >100 = data entry error
AGE_MIN, AGE_MAX = 1, 100

# Reporting delays beyond this are flagged (not removed) as suspicious
MAX_REASONABLE_DELAY_DAYS = 365

SEX_MAP = {
    "M": "Male", "F": "Female",
    "X": "Unknown", "H": "Unknown", "-": "Unknown",
}

DESCENT_MAP = {
    "W": "White", "B": "Black", "H": "Hispanic/Latino", "A": "Other Asian",
    "C": "Chinese", "K": "Korean", "J": "Japanese", "F": "Filipino",
    "V": "Vietnamese", "I": "American Indian/Alaska Native",
    "Z": "Asian Indian", "P": "Pacific Islander", "U": "Hawaiian",
    "D": "Cambodian", "L": "Laotian", "S": "Samoan", "G": "Guamanian",
    "O": "Other", "X": "Unknown",
}

# Ordered by specificity — more specific patterns must come FIRST
CRIME_CATEGORY_RULES = [
    ("Homicide",        ["HOMICIDE", "MURDER", "MANSLAUGHTER"]),
    ("Sex Offense",     ["RAPE", "SEXUAL PENETRATION", "LEWD", "INDECENT"]),
    ("Robbery",         ["ROBBERY"]),
    ("Assault/Battery", ["ASSAULT", "BATTERY", "SHOTS FIRED"]),
    ("Burglary",        ["BURGLARY"]),
    ("Vehicle Crime",   ["VEHICLE - STOLEN", "VEHICLE - ATTEMPT STOLEN", "THEFT FROM MOTOR"]),
    ("Theft",           ["THEFT", "SHOPLIFTING", "PICKPOCKET", "PURSE SNATCHING"]),
    ("Vandalism",       ["VANDALISM"]),
    ("Drug Offense",    ["DRUG", "NARCOTIC"]),
    ("Identity Theft",  ["IDENTITY"]),
    ("Fraud",           ["FRAUD", "EMBEZZLE", "BUNCO", "COUNTERFEIT"]),
    ("Trespassing",     ["TRESPASS"]),
    ("Child Abuse",     ["CHILD", "MINOR"]),
]

PREMISE_CATEGORY_RULES = [
    ("Residential",     ["SINGLE FAMILY DWELLING", "MULTI-UNIT DWELLING", "APARTMENT", "HOUSE", "RESIDENCE"]),
    ("Vehicle",         ["VEHICLE, PASSENGER/CAR", "VEHICLE", "AUTO"]),
    ("Public Street",   ["STREET", "SIDEWALK", "ALLEY", "HIGHWAY"]),
    ("Parking Area",    ["PARKING LOT", "PARKING GARAGE", "DRIVEWAY"]),
    ("Commercial",      ["STORE", "MARKET", "BUSINESS", "SHOP", "RESTAURANT", "BANK"]),
    ("Transit",         ["MTA", "BUS", "TRAIN", "METRO", "TRANSPORT", "SUBWAY"]),
    ("Park/Recreation", ["PARK", "PLAYGROUND", "BEACH", "RECREATION"]),
    ("Educational",     ["SCHOOL", "COLLEGE", "UNIVERSITY", "CAMPUS"]),
    ("Government",      ["GOVERNMENT", "CITY HALL", "COURT", "LIBRARY"]),
    ("Healthcare",      ["HOSPITAL", "CLINIC", "MEDICAL"]),
]

AGE_BINS   = [0,   12,  18,  25,  35,  45,  55,  65,  100]
AGE_LABELS = ["Child (0-11)", "Teen (12-17)", "Young Adult (18-24)",
              "Adult (25-34)", "Adult (35-44)", "Middle-Aged (45-54)",
              "Senior (55-64)", "Elderly (65+)"]


# ── Audit Trail ───────────────────────────────────────────────────────────────

class AuditTrail:
    """Tracks every cleaning decision with before/after row counts and change stats."""

    def __init__(self, total_rows: int):
        self.total_rows = total_rows
        self.steps: list[dict] = []

    def record(self, step: str, description: str, changed: int, detail: str = ""):
        pct = changed / self.total_rows * 100
        self.steps.append({
            "step": step,
            "description": description,
            "rows_affected": changed,
            "pct_affected": round(pct, 2),
            "detail": detail,
        })
        log.info(f"[{step}] {description} → {changed:,} rows affected ({pct:.1f}%) {detail}")

    def save(self, path: str):
        class _NumpyEncoder(json.JSONEncoder):
            """Convert numpy int/float types to native Python before serialising."""
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(path, "w") as f:
            json.dump({"total_rows": self.total_rows, "steps": self.steps}, f,
                      indent=2, cls=_NumpyEncoder)
        log.info(f"Audit trail saved → {path}")

    def summary(self):
        print("\n" + "=" * 65)
        print("CLEANING AUDIT SUMMARY")
        print("=" * 65)
        print(f"{'Step':<22} {'Affected':>10} {'%':>7}  Description")
        print("-" * 65)
        for s in self.steps:
            print(f"{s['step']:<22} {s['rows_affected']:>10,} {s['pct_affected']:>6.1f}%  {s['description']}")
        print("=" * 65)


# ── Helper: rule-based keyword categoriser ────────────────────────────────────

def _apply_keyword_rules(text: str, rules: list[tuple], default: str = "Other") -> str:
    """
    Match `text` against ordered keyword rules.
    First match wins — so put more-specific rules earlier in the list.
    """
    if pd.isna(text):
        return "Unknown"
    upper = str(text).upper()
    for category, keywords in rules:
        if any(kw in upper for kw in keywords):
            return category
    return default


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    log.info(f"Loading: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Immediately check for the columns we need
    required = {"DATE OCC", "TIME OCC", "Date Rptd", "Crm Cd Desc",
                 "Vict Age", "Vict Sex", "Vict Descent", "LAT", "LON"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    return df


# ── Step 2: Deduplicate ───────────────────────────────────────────────────────

def drop_duplicates(df: pd.DataFrame, audit: AuditTrail) -> pd.DataFrame:
    before = len(df)
    # DR_NO is the unique report number — exact duplicates on it are true dupes
    df = df.drop_duplicates(subset=["DR_NO"] if "DR_NO" in df.columns else None)
    removed = before - len(df)
    audit.record("Deduplication", "Exact duplicate rows removed", removed)
    return df


# ── Step 3: Parse Dates & Extract Temporal Features ──────────────────────────

def parse_dates(df: pd.DataFrame, audit: AuditTrail) -> pd.DataFrame:
    for col in ["Date Rptd", "DATE OCC"]:
        before_nulls = df[col].isna().sum()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        new_nulls = df[col].isna().sum() - before_nulls
        audit.record(f"Date parse: {col}", f"Unparseable values → NaT", new_nulls)

    # Temporal features from crime occurrence date
    df["Year"]      = df["DATE OCC"].dt.year
    df["Month"]     = df["DATE OCC"].dt.month
    df["MonthName"] = df["DATE OCC"].dt.strftime("%b")
    df["DayOfWeek"] = df["DATE OCC"].dt.day_name()
    df["IsWeekend"] = df["DATE OCC"].dt.dayofweek >= 5

    # TIME OCC is stored as HHMM integer (e.g. 1430 = 14:30)
    df["TIME OCC"]  = pd.to_numeric(df["TIME OCC"], errors="coerce").fillna(0).astype(int)
    df["TIME OCC"]  = df["TIME OCC"].clip(upper=2359)          # remove impossible values
    df["Hour"]      = df["TIME OCC"] // 100
    df["Minute"]    = df["TIME OCC"] % 100
    df["Hour"]      = df["Hour"].clip(0, 23)

    # Time-of-day bucket — useful for stratified analysis
    def _time_bucket(h):
        if 0  <= h < 6:  return "Night (12am–6am)"
        if 6  <= h < 12: return "Morning (6am–12pm)"
        if 12 <= h < 18: return "Afternoon (12pm–6pm)"
        return "Evening (6pm–12am)"

    df["TimeBucket"] = df["Hour"].apply(_time_bucket)

    audit.record("Temporal features", "Extracted 7 time features from dates", len(df), "(Year, Month, DayOfWeek, IsWeekend, Hour, Minute, TimeBucket)")
    return df


# ── Step 4: Reporting Delay ───────────────────────────────────────────────────

def compute_reporting_delay(df: pd.DataFrame, audit: AuditTrail) -> pd.DataFrame:
    df["Reporting Delay (Days)"] = (df["Date Rptd"] - df["DATE OCC"]).dt.days

    negative = (df["Reporting Delay (Days)"] < 0).sum()
    audit.record("Reporting delay", "Negative delays (date entry errors) clamped to 0", negative)
    df["Reporting Delay (Days)"] = df["Reporting Delay (Days)"].clip(lower=0)

    suspicious = (df["Reporting Delay (Days)"] > MAX_REASONABLE_DELAY_DAYS).sum()
    df["SuspiciousDelay"] = df["Reporting Delay (Days)"] > MAX_REASONABLE_DELAY_DAYS
    audit.record("Suspicious delay flag", f"Delays > {MAX_REASONABLE_DELAY_DAYS} days flagged", suspicious)

    # Bucket delays for easier grouping
    # Note: bins must be strictly increasing — "Same Day" (delay=0) handled via a separate map
    df["DelayBucket"] = pd.cut(
        df["Reporting Delay (Days)"].clip(upper=MAX_REASONABLE_DELAY_DAYS),
        bins=[0, 1, 7, 30, 90, MAX_REASONABLE_DELAY_DAYS],
        labels=["Same Day", "Within Week", "Within Month", "Within Quarter", "Over 90 Days"],
        include_lowest=True,
    )
    return df


# ── Step 5: Victim Age ────────────────────────────────────────────────────────

def clean_victim_age(df: pd.DataFrame, audit: AuditTrail) -> pd.DataFrame:
    # Age = 0 in LAPD data means "unknown", not actually zero years old
    zero_age = (df["Vict Age"] == 0).sum()
    df.loc[df["Vict Age"] == 0, "Vict Age"] = np.nan
    audit.record("Age: zeros → NaN", "LAPD encodes unknown age as 0", zero_age)

    # Ages outside human range are data entry errors
    out_of_range = ((df["Vict Age"] < AGE_MIN) | (df["Vict Age"] > AGE_MAX)).sum()
    df.loc[(df["Vict Age"] < AGE_MIN) | (df["Vict Age"] > AGE_MAX), "Vict Age"] = np.nan
    audit.record("Age: out-of-range → NaN", f"Ages outside [{AGE_MIN}, {AGE_MAX}] nulled", out_of_range)

    df["Age Group"] = pd.cut(
        df["Vict Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    )
    return df


# ── Step 6: Victim Sex ────────────────────────────────────────────────────────

def clean_victim_sex(df: pd.DataFrame, audit: AuditTrail) -> pd.DataFrame:
    unmapped_before = (~df["Vict Sex"].isin(SEX_MAP.keys()) & df["Vict Sex"].notna()).sum()
    df["Vict Sex"] = df["Vict Sex"].fillna("Unknown").str.strip().str.upper()
    df["Vict Sex Clean"] = df["Vict Sex"].map(SEX_MAP).fillna("Unknown")

    audit.record("Victim sex standardised", "Codes mapped to Male/Female/Unknown", unmapped_before)

    # Flag records where sex is unknown — useful for bias analysis
    df["Sex Unknown"] = df["Vict Sex Clean"] == "Unknown"
    return df


# ── Step 7: Victim Descent ────────────────────────────────────────────────────

def clean_victim_descent(df: pd.DataFrame, audit: AuditTrail) -> pd.DataFrame:
    df["Vict Descent"] = df["Vict Descent"].fillna("X").str.strip().str.upper()
    df["Vict Descent Clean"] = df["Vict Descent"].map(DESCENT_MAP).fillna("Unknown")

    unknown_count = (df["Vict Descent Clean"] == "Unknown").sum()
    audit.record("Victim descent mapped", "Single-letter codes → full descriptions", unknown_count,
                 f"({unknown_count/len(df)*100:.1f}% unknown — significant, track in EDA)")
    return df


# ── Step 8: Crime Categories ──────────────────────────────────────────────────

def categorise_crimes(df: pd.DataFrame, audit: AuditTrail) -> pd.DataFrame:
    df["Crime Category"] = df["Crm Cd Desc"].apply(
        lambda x: _apply_keyword_rules(x, CRIME_CATEGORY_RULES, default="Other")
    )
    other_count = (df["Crime Category"] == "Other").sum()
    audit.record("Crime categories", f"140+ types → 13 categories", other_count,
                 f"({other_count:,} fell into 'Other' — review CRIME_CATEGORY_RULES if high)")

    # Severity tier — useful as a target variable or filter
    high_severity    = {"Homicide", "Sex Offense", "Robbery", "Assault/Battery"}
    medium_severity  = {"Burglary", "Vehicle Crime", "Theft", "Child Abuse"}
    df["Severity"] = df["Crime Category"].apply(
        lambda c: "High" if c in high_severity
        else ("Medium" if c in medium_severity else "Low")
    )
    return df


# ── Step 9: Premise Categories ────────────────────────────────────────────────

def categorise_premises(df: pd.DataFrame, audit: AuditTrail) -> pd.DataFrame:
    if "Premis Desc" not in df.columns:
        log.warning("'Premis Desc' column not found — skipping premise categorisation")
        return df

    df["Premise Category"] = df["Premis Desc"].apply(
        lambda x: _apply_keyword_rules(x, PREMISE_CATEGORY_RULES, default="Other")
    )
    audit.record("Premise categories", "Premise descriptions grouped into 10 location types",
                 (df["Premise Category"] == "Other").sum())
    return df


# ── Step 10: Weapon Flag ──────────────────────────────────────────────────────

def flag_weapons(df: pd.DataFrame, audit: AuditTrail) -> pd.DataFrame:
    """
    Weapon Used Cd is 94% missing. Rather than analysing the sparse column,
    create a reliable binary flag and keep the descriptor for when it exists.
    """
    df["Has Weapon"]     = df["Weapon Used Cd"].notna()
    df["Weapon Desc"]    = df.get("Weapon Desc", pd.Series(dtype=str))

    with_weapon = df["Has Weapon"].sum()
    audit.record("Weapon flag created", "Binary Has_Weapon extracted from 94%-missing column",
                 with_weapon, f"({with_weapon/len(df)*100:.1f}% of crimes involve a weapon)")
    return df


# ── Step 11: Geographic Validation ───────────────────────────────────────────

def validate_coordinates(df: pd.DataFrame, audit: AuditTrail) -> pd.DataFrame:
    # (0, 0) is the ocean — LAPD uses it as a null coordinate
    zero_coords = ((df["LAT"] == 0) | (df["LON"] == 0)).sum()
    df.loc[(df["LAT"] == 0) | (df["LON"] == 0), ["LAT", "LON"]] = np.nan
    audit.record("Zero coordinates → NaN", "LAT/LON of (0,0) treated as missing", zero_coords)

    valid_mask = (
        df["LAT"].between(*LA_LAT) & df["LON"].between(*LA_LON)
    )
    outside_la = (~valid_mask & df["LAT"].notna()).sum()
    df["Valid Coordinates"] = valid_mask
    audit.record("Out-of-LA coords flagged", f"Outside bounding box {LA_LAT}, {LA_LON}", outside_la)
    return df


# ── Step 12: Drop Low-Value Columns ──────────────────────────────────────────

def drop_low_value_columns(df: pd.DataFrame, audit: AuditTrail) -> pd.DataFrame:
    """
    Remove columns that are >95% missing AND have no analytical value.
    We keep Weapon Used Cd description even though it's sparse — we extract Has_Weapon from it.
    """
    always_drop = ["Crm Cd 2", "Crm Cd 3", "Crm Cd 4"]
    cols_to_drop = [c for c in always_drop if c in df.columns]

    # Also auto-detect columns that are >98% empty (excluding ones we know we want)
    keep_sparse = {"Weapon Used Cd", "Weapon Desc"}
    for col in df.columns:
        if col in keep_sparse or col in cols_to_drop:
            continue
        miss_rate = df[col].isna().mean()
        if miss_rate > 0.98:
            cols_to_drop.append(col)

    df = df.drop(columns=cols_to_drop, errors="ignore")
    audit.record("Low-value columns dropped", f"Columns >98% missing removed", len(cols_to_drop),
                 f"({cols_to_drop})")
    return df


# ── Step 13: Unknown Demographic Bias Check ───────────────────────────────────

def compute_demographic_bias_flags(df: pd.DataFrame, audit: AuditTrail) -> pd.DataFrame:
    """
    The PPT notes 48.5% of victim data is unknown. This function checks whether
    unknowns are *randomly* distributed or concentrated in specific areas/crime types.
    Result is saved as a diagnostic column, not used to remove rows.
    """
    # Is victim info completely missing (sex AND descent both unknown)?
    df["Demographic Unknown"] = (
        (df["Vict Sex Clean"] == "Unknown") & (df["Vict Descent Clean"] == "Unknown")
    )
    fully_unknown = df["Demographic Unknown"].sum()
    audit.record("Demographic unknown flag", "Rows with ALL victim demographics unknown",
                 fully_unknown, f"({fully_unknown/len(df)*100:.1f}% — analyse distribution in EDA)")
    return df


# ── Pipeline Orchestrator ─────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    output_path: str,
    audit_path: str = "data/cleaning_audit.json",
) -> pd.DataFrame:
    """
    End-to-end cleaning pipeline. Call this to fully reproduce cleaned data.

    Parameters
    ----------
    input_path  : path to raw CSV from data.lacity.org
    output_path : path for cleaned CSV output
    audit_path  : path for JSON audit log (records every decision)

    Returns
    -------
    Cleaned DataFrame
    """
    log.info("=" * 60)
    log.info("LA CRIME DATA — CLEANING PIPELINE START")
    log.info("=" * 60)

    df = load_data(input_path)
    audit = AuditTrail(total_rows=len(df))

    # ── Ordered cleaning steps ────────────────────────────────────────────────
    df = drop_duplicates(df, audit)
    df = parse_dates(df, audit)
    df = compute_reporting_delay(df, audit)
    df = clean_victim_age(df, audit)
    df = clean_victim_sex(df, audit)
    df = clean_victim_descent(df, audit)
    df = categorise_crimes(df, audit)
    df = categorise_premises(df, audit)
    df = flag_weapons(df, audit)
    df = validate_coordinates(df, audit)
    df = drop_low_value_columns(df, audit)
    df = compute_demographic_bias_flags(df, audit)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(f"Cleaned data saved → {output_path}")
    log.info(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
    audit.save(audit_path)
    audit.summary()

    return df


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline(
        input_path="data/raw/crime_data_2024_to_present.csv",
        output_path="data/processed/crime_data_cleaned.csv",
        audit_path="data/cleaning_audit.json",
    )