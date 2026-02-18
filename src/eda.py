"""
Phase 1 Exploratory Data Analysis (EDA) pipeline.

Outputs:
1) CSV tables with major EDA results
2) PNG figures with labeled charts
3) JSON + TXT summary for report writing
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplcache"))

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "crime_data_cleaned.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "eda"

DATE_COLUMNS = ["date_occ", "date_rptd"]
NUMERIC_COLUMNS = [
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
    "lat",
    "lon",
    "vict_age",
    "occ_year",
    "occ_month",
    "report_delay_days",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run EDA pipeline on cleaned crime data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Cleaned CSV input path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="EDA output directory")
    return parser.parse_args()


def save_plot(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_table(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_cleaned_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def run_eda(input_path, output_dir):
    sns.set_theme(style="whitegrid")

    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_cleaned_data(input_path)
    operations: List[Dict[str, Any]] = []
    key_findings: List[str] = []

    def record_operation(
        name: str,
        description: str,
        tables = None,
        plots= None,
        status= "completed",
        note =None,
    ):
        entry: Dict[str, Any] = {
            "name": name,
            "description": description,
            "status": status,
            "tables": tables or [],
            "plots": plots or [],
        }
        if note:
            entry["note"] = note
        operations.append(entry)
    date_min = str(df["date_occ"].min().date()) if "date_occ" in df.columns and df["date_occ"].notna().any() else None
    date_max = str(df["date_occ"].max().date()) if "date_occ" in df.columns and df["date_occ"].notna().any() else None
    profile = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "unique_incidents_dr_no": int(df["dr_no"].nunique()) if "dr_no" in df.columns else None,
        "date_occ_min": date_min,
        "date_occ_max": date_max,
    }
    profile_df = pd.DataFrame([profile])
    profile_path = tables_dir / "01_dataset_profile.csv"
    save_table(profile_df, profile_path)
    record_operation(
        "dataset_profile",
        "Basic dataset shape, uniqueness, and date coverage.",
        tables=[str(profile_path)],
    )
    missing_df = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean() * 100).round(2).values,
        }
    ).sort_values(by="missing_pct", ascending=False)
    missing_path = tables_dir / "02_missingness_by_column.csv"
    save_table(missing_df, missing_path)

    missing_top = missing_df.head(15).sort_values("missing_pct", ascending=True)
    plt.figure(figsize=(11, 7))
    plt.barh(missing_top["column"], missing_top["missing_pct"], color="#2C7FB8")
    plt.xlabel("Missing Values (%)")
    plt.ylabel("Column")
    plt.title("Top 15 Columns by Missingness")
    missing_plot_path = plots_dir / "02_missingness_top15.png"
    save_plot(missing_plot_path)
    record_operation(
        "missingness_analysis",
        "Column-level missing-value audit with percentages.",
        tables=[str(missing_path)],
        plots=[str(missing_plot_path)],
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_summary = (
            df[numeric_cols]
            .describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
            .T.reset_index()
            .rename(columns={"index": "column"})
        )
        numeric_summary_path = tables_dir / "03_numeric_descriptive_stats.csv"
        save_table(numeric_summary, numeric_summary_path)
        record_operation(
            "numeric_descriptive_statistics",
            "Distribution summary for all numeric columns.",
            tables=[str(numeric_summary_path)],
        )
    else:
        record_operation(
            "numeric_descriptive_statistics",
            "Distribution summary for all numeric columns.",
            status="skipped",
            note="No numeric columns available.",
        )
    crime_col = "crm_cd_desc" if "crm_cd_desc" in df.columns else ("crm_cd" if "crm_cd" in df.columns else None)
    if crime_col:
        crime_counts = (
            df[crime_col]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
            .head(20)
            .rename_axis("crime_type")
            .reset_index(name="incident_count")
        )
        crime_counts["incident_pct"] = (crime_counts["incident_count"] / len(df) * 100).round(2)
        crime_table_path = tables_dir / "04_top20_crime_types.csv"
        save_table(crime_counts, crime_table_path)

        plot_df = crime_counts.sort_values("incident_count", ascending=True)
        plt.figure(figsize=(12, 8))
        plt.barh(plot_df["crime_type"], plot_df["incident_count"], color="#F28E2B")
        plt.xlabel("Incident Count")
        plt.ylabel("Crime Type")
        plt.title("Top 20 Crime Types")
        crime_plot_path = plots_dir / "04_top20_crime_types.png"
        save_plot(crime_plot_path)

        top_crime = crime_counts.iloc[0]
        key_findings.append(
            f"Most common crime type: {top_crime['crime_type']} ({int(top_crime['incident_count'])} incidents, {top_crime['incident_pct']}%)."
        )
        record_operation(
            "crime_type_frequency",
            "Top crime categories by count and percentage.",
            tables=[str(crime_table_path)],
            plots=[str(crime_plot_path)],
        )
    else:
        record_operation(
            "crime_type_frequency",
            "Top crime categories by count and percentage.",
            status="skipped",
            note="No crime description/code column found.",
        )
        
    if "area_name" in df.columns:
        area_counts = (
            df["area_name"]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
            .head(20)
            .rename_axis("area_name")
            .reset_index(name="incident_count")
        )
        area_counts["incident_pct"] = (area_counts["incident_count"] / len(df) * 100).round(2)
        area_table_path = tables_dir / "05_top20_areas.csv"
        save_table(area_counts, area_table_path)

        area_plot_df = area_counts.sort_values("incident_count", ascending=True)
        plt.figure(figsize=(11, 8))
        plt.barh(area_plot_df["area_name"], area_plot_df["incident_count"], color="#59A14F")
        plt.xlabel("Incident Count")
        plt.ylabel("Area")
        plt.title("Top 20 Areas by Incident Volume")
        area_plot_path = plots_dir / "05_top20_areas.png"
        save_plot(area_plot_path)

        top_area = area_counts.iloc[0]
        key_findings.append(
            f"Highest incident area: {top_area['area_name']} ({int(top_area['incident_count'])} incidents, {top_area['incident_pct']}%)."
        )
        record_operation(
            "area_distribution",
            "Incident concentration by policing area.",
            tables=[str(area_table_path)],
            plots=[str(area_plot_path)],
        )
    else:
        record_operation(
            "area_distribution",
            "Incident concentration by policing area.",
            status="skipped",
            note="Column `area_name` not found.",
        )
    if "date_occ" in df.columns and df["date_occ"].notna().any():
        month_series = df["date_occ"].dt.to_period("M").dt.to_timestamp()
        monthly_counts = (
            month_series.value_counts().sort_index().rename_axis("month").reset_index(name="incident_count")
        )
        monthly_table_path = tables_dir / "06_monthly_trend.csv"
        save_table(monthly_counts, monthly_table_path)

        plt.figure(figsize=(12, 5))
        plt.plot(monthly_counts["month"], monthly_counts["incident_count"], marker="o", color="#4E79A7")
        plt.xlabel("Month")
        plt.ylabel("Incident Count")
        plt.title("Monthly Crime Trend")
        plt.xticks(rotation=45)
        monthly_plot_path = plots_dir / "06_monthly_trend.png"
        save_plot(monthly_plot_path)
        record_operation(
            "monthly_trend",
            "Temporal trend of incidents aggregated by month.",
            tables=[str(monthly_table_path)],
            plots=[str(monthly_plot_path)],
        )
    else:
        record_operation(
            "monthly_trend",
            "Temporal trend of incidents aggregated by month.",
            status="skipped",
            note="Column `date_occ` missing or empty.",
        )

    dow_col = "occ_day_of_week"
    if dow_col not in df.columns and "date_occ" in df.columns:
        df[dow_col] = df["date_occ"].dt.day_name()

    if dow_col in df.columns:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = (
            df[dow_col]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
            .reindex(day_order + ["UNKNOWN"], fill_value=0)
            .rename_axis("day_of_week")
            .reset_index(name="incident_count")
        )
        dow_table_path = tables_dir / "07_day_of_week_distribution.csv"
        save_table(dow_counts, dow_table_path)

        plot_dow = dow_counts[dow_counts["day_of_week"] != "UNKNOWN"]
        plt.figure(figsize=(11, 5))
        plt.bar(plot_dow["day_of_week"], plot_dow["incident_count"], color="#E15759")
        plt.xlabel("Day of Week")
        plt.ylabel("Incident Count")
        plt.title("Incidents by Day of Week")
        plt.xticks(rotation=25)
        dow_plot_path = plots_dir / "07_day_of_week_distribution.png"
        save_plot(dow_plot_path)

        peak_day = plot_dow.sort_values("incident_count", ascending=False).iloc[0]
        key_findings.append(
            f"Peak day of week: {peak_day['day_of_week']} ({int(peak_day['incident_count'])} incidents)."
        )
        record_operation(
            "day_of_week_pattern",
            "Distribution of incidents by day of week.",
            tables=[str(dow_table_path)],
            plots=[str(dow_plot_path)],
        )
    else:
        record_operation(
            "day_of_week_pattern",
            "Distribution of incidents by day of week.",
            status="skipped",
            note="No day-of-week information available.",
        )

    if "occ_hour" not in df.columns and "time_occ" in df.columns:
        time_numeric = pd.to_numeric(df["time_occ"], errors="coerce")
        df["occ_hour"] = np.floor_divide(time_numeric, 100)
        df.loc[~df["occ_hour"].between(0, 23), "occ_hour"] = np.nan

    if "occ_hour" in df.columns:
        hour_counts = (
            df["occ_hour"]
            .dropna()
            .astype(int)
            .value_counts()
            .reindex(range(24), fill_value=0)
            .rename_axis("hour")
            .reset_index(name="incident_count")
        )
        hour_table_path = tables_dir / "08_hour_of_day_distribution.csv"
        save_table(hour_counts, hour_table_path)

        plt.figure(figsize=(12, 5))
        plt.plot(hour_counts["hour"], hour_counts["incident_count"], marker="o", color="#76B7B2")
        plt.xlabel("Hour of Day (0-23)")
        plt.ylabel("Incident Count")
        plt.title("Incidents by Hour of Day")
        plt.xticks(range(0, 24, 1))
        hour_plot_path = plots_dir / "08_hour_of_day_distribution.png"
        save_plot(hour_plot_path)

        peak_hour = hour_counts.sort_values("incident_count", ascending=False).iloc[0]
        key_findings.append(f"Peak incident hour: {int(peak_hour['hour']):02d}:00 ({int(peak_hour['incident_count'])} incidents).")
        record_operation(
            "hour_of_day_pattern",
            "Hourly crime pattern across a 24-hour cycle.",
            tables=[str(hour_table_path)],
            plots=[str(hour_plot_path)],
        )
    else:
        record_operation(
            "hour_of_day_pattern",
            "Hourly crime pattern across a 24-hour cycle.",
            status="skipped",
            note="No usable occurrence-time information available.",
        )

    # 9) Victim age distribution
    if "vict_age" in df.columns and df["vict_age"].notna().any():
        age_series = df["vict_age"].dropna()
        age_summary = pd.DataFrame(
            [
                {
                    "count": int(age_series.count()),
                    "mean": round(float(age_series.mean()), 2),
                    "median": round(float(age_series.median()), 2),
                    "std": round(float(age_series.std()), 2),
                    "min": int(age_series.min()),
                    "q1": round(float(age_series.quantile(0.25)), 2),
                    "q3": round(float(age_series.quantile(0.75)), 2),
                    "max": int(age_series.max()),
                }
            ]
        )
        age_summary_path = tables_dir / "09_victim_age_summary.csv"
        save_table(age_summary, age_summary_path)

        age_bins = [0, 12, 17, 24, 34, 44, 54, 64, 120]
        age_labels = ["0-12", "13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        age_bucket = pd.cut(age_series, bins=age_bins, labels=age_labels, right=True)
        age_bucket_df = age_bucket.value_counts().sort_index().rename_axis("age_group").reset_index(name="incident_count")
        age_bucket_path = tables_dir / "09_victim_age_groups.csv"
        save_table(age_bucket_df, age_bucket_path)

        plt.figure(figsize=(11, 5))
        sns.histplot(age_series, bins=30, color="#AF7AA1")
        plt.xlabel("Victim Age")
        plt.ylabel("Incident Count")
        plt.title("Victim Age Distribution")
        age_plot_path = plots_dir / "09_victim_age_distribution.png"
        save_plot(age_plot_path)

        key_findings.append(
            f"Victim age median: {round(float(age_series.median()), 2)} years (mean {round(float(age_series.mean()), 2)})."
        )
        record_operation(
            "victim_age_distribution",
            "Victim age summary statistics, bins, and histogram.",
            tables=[str(age_summary_path), str(age_bucket_path)],
            plots=[str(age_plot_path)],
        )
    else:
        record_operation(
            "victim_age_distribution",
            "Victim age summary statistics, bins, and histogram.",
            status="skipped",
            note="Column `vict_age` missing or fully empty.",
        )
        
    if "vict_sex" in df.columns:
        sex_counts = (
            df["vict_sex"]
            .fillna("X")
            .astype(str)
            .str.upper()
            .replace("", "X")
            .value_counts()
            .rename_axis("vict_sex")
            .reset_index(name="incident_count")
        )
        sex_counts["incident_pct"] = (sex_counts["incident_count"] / len(df) * 100).round(2)
        sex_table_path = tables_dir / "10_victim_sex_distribution.csv"
        save_table(sex_counts, sex_table_path)

        plt.figure(figsize=(8, 5))
        plt.bar(sex_counts["vict_sex"], sex_counts["incident_count"], color="#FF9DA7")
        plt.xlabel("Victim Sex")
        plt.ylabel("Incident Count")
        plt.title("Victim Sex Distribution")
        sex_plot_path = plots_dir / "10_victim_sex_distribution.png"
        save_plot(sex_plot_path)
        record_operation(
            "victim_sex_distribution",
            "Victim sex category distribution.",
            tables=[str(sex_table_path)],
            plots=[str(sex_plot_path)],
        )
    else:
        record_operation(
            "victim_sex_distribution",
            "Victim sex category distribution.",
            status="skipped",
            note="Column `vict_sex` not found.",
        )

    if "weapon_desc" in df.columns:
        weapon_counts = (
            df["weapon_desc"]
            .fillna("NO WEAPON REPORTED")
            .astype(str)
            .str.upper()
            .value_counts()
            .head(20)
            .rename_axis("weapon_desc")
            .reset_index(name="incident_count")
        )
        weapon_counts["incident_pct"] = (weapon_counts["incident_count"] / len(df) * 100).round(2)
        weapon_table_path = tables_dir / "11_weapon_usage_top20.csv"
        save_table(weapon_counts, weapon_table_path)

        weapon_plot_df = weapon_counts.sort_values("incident_count", ascending=True)
        plt.figure(figsize=(12, 8))
        plt.barh(weapon_plot_df["weapon_desc"], weapon_plot_df["incident_count"], color="#9C755F")
        plt.xlabel("Incident Count")
        plt.ylabel("Weapon Category")
        plt.title("Top 20 Weapon Usage Categories")
        weapon_plot_path = plots_dir / "11_weapon_usage_top20.png"
        save_plot(weapon_plot_path)
        record_operation(
            "weapon_usage_analysis",
            "Weapon category frequencies, including non-weapon incidents.",
            tables=[str(weapon_table_path)],
            plots=[str(weapon_plot_path)],
        )
    else:
        record_operation(
            "weapon_usage_analysis",
            "Weapon category frequencies, including non-weapon incidents.",
            status="skipped",
            note="Column `weapon_desc` not found.",
        )
        
    if "report_delay_days" not in df.columns and {"date_rptd", "date_occ"}.issubset(df.columns):
        df["report_delay_days"] = (df["date_rptd"] - df["date_occ"]).dt.days
        df.loc[df["report_delay_days"] < 0, "report_delay_days"] = np.nan

    if "report_delay_days" in df.columns and df["report_delay_days"].notna().any():
        delay = df["report_delay_days"].dropna()
        delay_summary = pd.DataFrame(
            [
                {
                    "count": int(delay.count()),
                    "mean": round(float(delay.mean()), 2),
                    "median": round(float(delay.median()), 2),
                    "p90": round(float(delay.quantile(0.9)), 2),
                    "p95": round(float(delay.quantile(0.95)), 2),
                    "max": round(float(delay.max()), 2),
                }
            ]
        )
        delay_summary_path = tables_dir / "12_report_delay_summary.csv"
        save_table(delay_summary, delay_summary_path)

        capped_delay = delay.clip(upper=delay.quantile(0.95))
        plt.figure(figsize=(11, 5))
        sns.histplot(capped_delay, bins=30, color="#BAB0AC")
        plt.xlabel("Report Delay (Days, capped at 95th percentile)")
        plt.ylabel("Incident Count")
        plt.title("Distribution of Report Delay")
        delay_plot_path = plots_dir / "12_report_delay_distribution.png"
        save_plot(delay_plot_path)

        key_findings.append(
            f"Median report delay: {round(float(delay.median()), 2)} days (95th percentile: {round(float(delay.quantile(0.95)), 2)} days)."
        )
        record_operation(
            "report_delay_analysis",
            "Summary and distribution of delay between occurrence and reporting.",
            tables=[str(delay_summary_path)],
            plots=[str(delay_plot_path)],
        )
    else:
        record_operation(
            "report_delay_analysis",
            "Summary and distribution of delay between occurrence and reporting.",
            status="skipped",
            note="No usable report delay information found.",
        )

    status_col = "status_desc" if "status_desc" in df.columns else ("status" if "status" in df.columns else None)
    if status_col:
        status_counts = (
            df[status_col]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
            .value_counts()
            .rename_axis("status")
            .reset_index(name="incident_count")
        )
        status_counts["incident_pct"] = (status_counts["incident_count"] / len(df) * 100).round(2)
        status_table_path = tables_dir / "13_case_status_distribution.csv"
        save_table(status_counts, status_table_path)

        top_status = status_counts.head(10).sort_values("incident_count", ascending=True)
        plt.figure(figsize=(10, 6))
        plt.barh(top_status["status"], top_status["incident_count"], color="#EDC948")
        plt.xlabel("Incident Count")
        plt.ylabel("Case Status")
        plt.title("Top Case Status Categories")
        status_plot_path = plots_dir / "13_case_status_distribution.png"
        save_plot(status_plot_path)
        record_operation(
            "case_status_distribution",
            "Distribution of reported case statuses.",
            tables=[str(status_table_path)],
            plots=[str(status_plot_path)],
        )
    else:
        record_operation(
            "case_status_distribution",
            "Distribution of reported case statuses.",
            status="skipped",
            note="No case status column found.",
        )
        
        
        
    if {"lat", "lon"}.issubset(df.columns):
        geo_df = df[["lon", "lat"]].dropna()
        if not geo_df.empty:
            plt.figure(figsize=(10, 8))
            hb = plt.hexbin(geo_df["lon"], geo_df["lat"], gridsize=65, cmap="viridis", mincnt=1)
            cb = plt.colorbar(hb)
            cb.set_label("Incident Count")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Geospatial Hotspots of Incidents (Hexbin)")
            geo_plot_path = plots_dir / "14_geospatial_hotspots_hexbin.png"
            save_plot(geo_plot_path)
            record_operation(
                "geospatial_hotspot_analysis",
                "Hexbin map showing geographic concentration of incidents.",
                plots=[str(geo_plot_path)],
            )
        else:
            record_operation(
                "geospatial_hotspot_analysis",
                "Hexbin map showing geographic concentration of incidents.",
                status="skipped",
                note="No valid latitude/longitude values available.",
            )
    else:
        record_operation(
            "geospatial_hotspot_analysis",
            "Hexbin map showing geographic concentration of incidents.",
            status="skipped",
            note="`lat`/`lon` columns not found.",
        )
    completed_ops = [op for op in operations if op["status"] == "completed"]
    skipped_ops = [op for op in operations if op["status"] == "skipped"]
    eda_summary: Dict[str, Any] = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "rows_analyzed": int(len(df)),
        "columns_analyzed": int(len(df.columns)),
        "completed_operations": len(completed_ops),
        "skipped_operations": len(skipped_ops),
        "operations": operations,
        "key_findings": key_findings,
    }

    summary_json_path = output_dir / "eda_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2, default=str)

    findings_txt_path = output_dir / "eda_key_findings.txt"
    with findings_txt_path.open("w", encoding="utf-8") as f:
        f.write("EDA Key Findings\n")
        f.write("=================\n")
        if key_findings:
            for idx, finding in enumerate(key_findings, start=1):
                f.write(f"{idx}. {finding}\n")
        else:
            f.write("No key findings generated.\n")

    return {
        "summary_json": str(summary_json_path),
        "findings_txt": str(findings_txt_path),
        "tables_dir": str(tables_dir),
        "plots_dir": str(plots_dir),
        "completed_operations": len(completed_ops),
        "skipped_operations": len(skipped_ops),
    }

def main():
    args = parse_args()
    result = run_eda(args.input, args.output_dir)
    print("EDA complete.")
    print(f"Completed operations: {result['completed_operations']}")
    print(f"Skipped operations: {result['skipped_operations']}")
    print(f"Tables: {result['tables_dir']}")
    print(f"Plots: {result['plots_dir']}")
    print(f"Summary JSON: {result['summary_json']}")
    print(f"Key findings TXT: {result['findings_txt']}")


if __name__ == "__main__":
    main()
