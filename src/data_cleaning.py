"""
Phase 1 data cleaning pipeline for LA crime dataset.

This script performs a reproducible sequence of cleaning operations and writes:
1) cleaned CSV dataset
2) JSON cleaning summary (step-level metrics)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "crime_data_2024_to_present.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "crime_data_cleaned.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "data_cleaning_summary.json"


def normalize_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def clean_object_columns(df):
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns
    if len(object_cols) == 0:
        return df

    placeholder_values = {
        "": pd.NA,
        " ": pd.NA,
        "na": pd.NA,
        "n/a": pd.NA,
        "null": pd.NA,
        "none": pd.NA,
        "unknown": pd.NA,
    }
    for col in object_cols:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .replace(placeholder_values, regex=False)
        )
    return df


def coerce_numeric_columns(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_datetime_columns(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_occurrence_time(df):
    df = df.copy()
    if "time_occ" not in df.columns:
        return df

    df["time_occ"] = pd.to_numeric(df["time_occ"], errors="coerce")
    df.loc[(df["time_occ"] < 0) | (df["time_occ"] > 2359), "time_occ"] = pd.NA

    occ_hour = (df["time_occ"] // 100).astype("Int64")
    occ_minute = (df["time_occ"] % 100).astype("Int64")
    invalid_minutes = occ_minute > 59
    occ_hour = occ_hour.mask(invalid_minutes)
    occ_minute = occ_minute.mask(invalid_minutes)

    df["occ_hour"] = occ_hour
    df["occ_minute"] = occ_minute
    return df


def clean_victim_age(df):
    df = df.copy()
    if "vict_age" not in df.columns:
        return df

    df["vict_age"] = pd.to_numeric(df["vict_age"], errors="coerce")
    invalid_age = (df["vict_age"] <= 0) | (df["vict_age"] > 100)
    df.loc[invalid_age, "vict_age"] = pd.NA
    return df


def norm_cat_cols(df):
    df = df.copy()
    for col in ["vict_sex", "vict_descent", "status", "status_desc", "area_name"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.upper().str.strip()

    if "vict_sex" in df.columns:
        df["vict_sex"] = df["vict_sex"].where(df["vict_sex"].isin(["M", "F", "X"]), "X")
        df["vict_sex"] = df["vict_sex"].fillna("X")

    if "vict_descent" in df.columns:
        df["vict_descent"] = df["vict_descent"].fillna("X")

    if "status" in df.columns:
        df["status"] = df["status"].fillna("UNK")
    if "status_desc" in df.columns:
        df["status_desc"] = df["status_desc"].fillna("UNKNOWN")
    return df


def validate_cods(df):
    df = df.copy()
    if "lat" not in df.columns or "lon" not in df.columns:
        return df

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    both_present = df["lat"].notna() & df["lon"].notna()
    zero_coords = both_present & (df["lat"] == 0) & (df["lon"] == 0)
    # Approx LA County bounds for sanity filtering.
    out_of_bounds = both_present & (
        (~df["lat"].between(33.5, 34.4)) | (~df["lon"].between(-118.95, -117.6))
    )

    invalid_coords = zero_coords | out_of_bounds
    df.loc[invalid_coords, ["lat", "lon"]] = pd.NA
    return df


def add_derived_cols(df):
    df = df.copy()
    if "date_occ" in df.columns:
        df["occ_year"] = df["date_occ"].dt.year
        df["occ_month"] = df["date_occ"].dt.month
        df["occ_day_of_week"] = df["date_occ"].dt.day_name()

    if "date_rptd" in df.columns and "date_occ" in df.columns:
        delay = (df["date_rptd"] - df["date_occ"]).dt.days
        delay = delay.mask(delay < 0, pd.NA)
        df["report_delay_days"] = delay
    return df


def drop_sparse_columns(df, threshold = 0.98):
    df = df.copy()
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio >= threshold].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df, cols_to_drop


def drop_rows_mis_req_flds(df):
    df = df.copy()
    required = [col for col in ["date_occ", "crm_cd", "area"] if col in df.columns]
    if required:
        df = df.dropna(subset=required)
    return df


def cast_null_int_cols(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def run_cleaning_pipeline(input_path, output_path, report_path):
    if not input_path.exists():
        raise FileNotFoundError(f"Raw input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    steps: List[Dict[str, object]] = []

    def track_step(step_name: str, before_rows: int, after_rows: int, **details: object) -> None:
        step_record: Dict[str, object] = {
            "step": step_name,
            "before_rows": before_rows,
            "after_rows": after_rows,
            "rows_changed": before_rows - after_rows,
        }
        step_record.update(details)
        steps.append(step_record)

    df = pd.read_csv(input_path, low_memory=False)
    initial_rows = len(df)
    initial_columns = list(df.columns)

    '''Standardise colm names'''
    before = len(df)
    df = normalize_column_names(df)
    track_step("normalize_column_names", before, len(df), columns=len(df.columns))

    '''Trim object columns and missing markers'''
    before = len(df)
    df = clean_object_columns(df)
    track_step("clean_object_columns", before, len(df))
    before = len(df)
    df = coerce_numeric_columns(
        df,
        [
            "dr_no",
            "time_occ",
            "area",
            "rpt_dist_no",
            "part_1_2",
            "crm_cd",
            "premis_cd",
            "weapon_used_cd",
            "crm_cd_1",
            "crm_cd_2",
            "crm_cd_3",
            "crm_cd_4",
            "lat",
            "lon",
            "vict_age",
        ],
    )
    track_step("coerce_numeric_columns", before, len(df))

    before = len(df)
    df = parse_datetime_columns(df, ["date_occ", "date_rptd"])
    track_step("parse_datetime_columns", before, len(df))

    before = len(df)
    df = clean_occurrence_time(df)
    track_step("clean_occurrence_time", before, len(df))

    before = len(df)
    df = df.drop_duplicates()
    track_step("drop_exact_duplicates", before, len(df))
    before = len(df)
    if {"dr_no", "date_rptd"}.issubset(df.columns):
        df = df.sort_values("date_rptd").drop_duplicates(subset=["dr_no"], keep="last")
    elif "dr_no" in df.columns:
        df = df.drop_duplicates(subset=["dr_no"], keep="first")
    track_step("drop_duplicate_incidents_by_dr_no", before, len(df))

    before = len(df)
    df = clean_victim_age(df)
    track_step("clean_victim_age", before, len(df))
    
    before = len(df)
    df = norm_cat_cols(df)
    track_step("normalize_category_columns", before, len(df))

    before = len(df)
    df = validate_cods(df)
    track_step("validate_coordinates", before, len(df))

    before = len(df)
    df = add_derived_cols(df)
    track_step("add_derived_columns", before, len(df), columns=len(df.columns))
    before = len(df)
    df, dropped_sparse_columns = drop_sparse_columns(df, threshold=0.98)
    track_step(
        "drop_sparse_columns",
        before,
        len(df),
        dropped_columns=dropped_sparse_columns,
        dropped_count=len(dropped_sparse_columns),
    )

    before = len(df)
    df = drop_rows_mis_req_flds(df)
    track_step("drop_rows_missing_required_fields", before, len(df))

    before = len(df)
    df = cast_null_int_cols(
        df,
        [
            "dr_no",
            "time_occ",
            "occ_hour",
            "occ_minute",
            "area",
            "rpt_dist_no",
            "part_1_2",
            "crm_cd",
            "premis_cd",
            "weapon_used_cd",
            "crm_cd_1",
            "crm_cd_2",
            "crm_cd_3",
            "crm_cd_4",
            "vict_age",
            "occ_year",
            "occ_month",
            "report_delay_days",
        ],
    )
    track_step("cast_nullable_integer_columns", before, len(df))

    before = len(df)
    sort_cols = [c for c in ["date_occ", "dr_no"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    track_step("sort_and_reset_index", before, len(df))

    df.to_csv(output_path, index=False)

    summary: Dict[str, object] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "rows_removed": initial_rows - len(df),
        "initial_columns": initial_columns,
        "final_columns": list(df.columns),
        "final_column_count": len(df.columns),
        "steps": steps,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run data cleaning pipeline.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Raw CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Cleaned CSV path")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="JSON cleaning summary path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_cleaning_pipeline(args.input, args.output, args.report)
    print(f"Cleaning complete: {summary['final_rows']} rows, {summary['final_column_count']} columns")
    print(f"Cleaned dataset: {summary['output_path']}")
    print(f"Summary report: {args.report}")


if __name__ == "__main__":
    main()
