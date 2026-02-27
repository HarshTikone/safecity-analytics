"""
eda.py
Production-Grade Exploratory Data Analysis for LA Crime Data
EAS 587 - Phase 1 Project

Design principles:
- Every plot answers a specific question stated in the PPT
- Visuals are publication-ready (labeled, titled, sourced)
- The demographic bias problem (48.5% unknown) is explicitly surfaced
- The stated hypothesis is directly tested with data
- All outputs are reproducible and saved with descriptive names
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE  = "YlOrRd"
ACCENT   = "#D62728"   # red — draws attention to key findings
NEUTRAL  = "#4C72B0"   # blue — standard bars
BG_GRAY  = "#F7F7F7"
FIG_DIR  = Path("data/processed/eda/plots")

plt.rcParams.update({
    "figure.facecolor": BG_GRAY,
    "axes.facecolor":   BG_GRAY,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.labelsize":   11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "font.family":      "sans-serif",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {path}")


def _source_note(ax, note="Source: LAPD Open Data / data.lacity.org"):
    ax.annotate(note, xy=(0, -0.12), xycoords="axes fraction",
                fontsize=7, color="gray")


def fmt_thousands(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def load_cleaned_data(filepath: str) -> pd.DataFrame:
    print(f"Loading cleaned data from: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["Date Rptd", "DATE OCC"], low_memory=False)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns\n")
    return df


# ── EDA 1: Data Quality Summary ───────────────────────────────────────────────

def eda_data_quality(df: pd.DataFrame):
    """
    Q: What is the actual state of the data before analysis?
    Surfaces missingness and the demographic unknown problem explicitly.
    """
    print("=" * 60)
    print("EDA 1 | DATA QUALITY OVERVIEW")
    print("=" * 60)

    cols_of_interest = [
        "Vict Age", "Vict Sex Clean", "Vict Descent Clean",
        "LAT", "LON", "Weapon Used Cd", "Premis Desc",
        "Reporting Delay (Days)", "Has Weapon",
    ]
    existing = [c for c in cols_of_interest if c in df.columns]
    miss = df[existing].isna().mean().sort_values(ascending=False) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Data Quality: Missingness & Demographic Unknowns", fontsize=14, fontweight="bold")

    # Missing rate bar chart
    colors = [ACCENT if v > 40 else NEUTRAL for v in miss.values]
    axes[0].barh(miss.index, miss.values, color=colors)
    axes[0].set_xlabel("% Missing")
    axes[0].set_title("Missing Data by Column")
    axes[0].axvline(40, color=ACCENT, linestyle="--", alpha=0.5, label="40% threshold")
    for i, v in enumerate(miss.values):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    axes[0].legend(fontsize=8)

    # Demographic unknowns by AREA — is the missingness random or biased?
    if "Demographic Unknown" in df.columns and "AREA NAME" in df.columns:
        unknown_by_area = (
            df.groupby("AREA NAME")["Demographic Unknown"]
            .mean()
            .sort_values(ascending=False) * 100
        )
        colors2 = [ACCENT if v > 50 else NEUTRAL for v in unknown_by_area.values]
        axes[1].barh(unknown_by_area.index, unknown_by_area.values, color=colors2)
        axes[1].set_xlabel("% Records with ALL Demographics Unknown")
        axes[1].set_title("Demographic Unknown Rate by LAPD Area\n(High = biased missing, not random)")
        axes[1].axvline(unknown_by_area.mean(), color="green", linestyle="--",
                        label=f"Avg: {unknown_by_area.mean():.1f}%")
        axes[1].legend(fontsize=8)
        _source_note(axes[1])

    plt.tight_layout()
    _save(fig, "01_data_quality")

    print(f"  Weapon Used Cd: {df['Weapon Used Cd'].isna().mean()*100:.1f}% missing")
    if "Demographic Unknown" in df.columns:
        print(f"  Fully unknown victim demographics: {df['Demographic Unknown'].sum():,} "
              f"({df['Demographic Unknown'].mean()*100:.1f}%)")


# ── EDA 2: Crime Volume Over Time ─────────────────────────────────────────────

def eda_temporal_trends(df: pd.DataFrame):
    """
    Q: Are crime rates stable, rising, or falling across 2024?
    Q: What time of day and day of week do crimes peak?
    """
    print("\n" + "=" * 60)
    print("EDA 2 | TEMPORAL TRENDS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Temporal Crime Patterns", fontsize=14, fontweight="bold")

    # Monthly trend with a smoothed line
    monthly = df.groupby("Month").size()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax = axes[0, 0]
    ax.bar(monthly.index, monthly.values, color=NEUTRAL, alpha=0.6, label="Monthly count")
    ax.plot(monthly.index, monthly.values, marker="o", color=ACCENT, linewidth=2, label="Trend")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, rotation=45)
    ax.set_title("Monthly Crime Volume (2024)")
    ax.set_ylabel("Number of Crimes")
    fmt_thousands(ax)
    ax.legend(fontsize=8)

    # Hour-of-day heatmap by crime category (top 6)
    ax = axes[0, 1]
    top_cats = df["Crime Category"].value_counts().head(6).index
    hourly = (
        df[df["Crime Category"].isin(top_cats)]
        .groupby(["Crime Category", "Hour"])
        .size()
        .unstack(fill_value=0)
    )
    # Normalise each row so we see *when* each crime type peaks, not which is most common
    hourly_norm = hourly.div(hourly.max(axis=1), axis=0)
    sns.heatmap(hourly_norm, ax=ax, cmap=PALETTE, cbar_kws={"label": "Relative Frequency"},
                linewidths=0.3)
    ax.set_title("Crime Type vs Hour (Normalised)\nWhen does each type peak?")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("")

    # Day-of-week pattern
    ax = axes[1, 0]
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    daily = df["DayOfWeek"].value_counts().reindex(day_order)
    bar_colors = [ACCENT if d in ("Friday","Saturday","Sunday") else NEUTRAL for d in day_order]
    ax.bar(range(7), daily.values, color=bar_colors)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax.set_title("Crimes by Day of Week\n(Weekend days highlighted)")
    ax.set_ylabel("Number of Crimes")
    fmt_thousands(ax)
    _source_note(ax)

    # Weekday vs Weekend box plot by TimeBucket
    ax = axes[1, 1]
    if "TimeBucket" in df.columns:
        bucket_order = ["Night (12am–6am)","Morning (6am–12pm)","Afternoon (12pm–6pm)","Evening (6pm–12am)"]
        bucket_counts = (
            df.groupby(["TimeBucket", "IsWeekend"])
            .size()
            .reset_index(name="count")
        )
        bucket_counts["Type"] = bucket_counts["IsWeekend"].map({True: "Weekend", False: "Weekday"})
        for label, grp in bucket_counts.groupby("Type"):
            ordered = grp.set_index("TimeBucket").reindex(bucket_order)["count"]
            ax.plot(range(4), ordered.values, marker="o", label=label, linewidth=2)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Night","Morning","Afternoon","Evening"], rotation=15)
        ax.set_title("Crime Volume: Weekday vs Weekend by Time Slot")
        ax.set_ylabel("Number of Crimes")
        ax.legend()
        fmt_thousands(ax)

    plt.tight_layout()
    _save(fig, "02_temporal_trends")

    peak_hour = df["Hour"].value_counts().idxmax()
    print(f"  Peak crime hour: {peak_hour}:00")
    print(f"  Highest-crime day: {df['DayOfWeek'].value_counts().idxmax()}")
    print(f"  Highest-crime month: {month_labels[df['Month'].value_counts().idxmax()-1]}")


# ── EDA 3: Geographic Distribution ───────────────────────────────────────────

def eda_geographic(df: pd.DataFrame):
    """
    Q: Which LAPD areas have the most crime and what types dominate there?
    This replaces a raw scatter plot (colored by crime code) with something readable.
    """
    print("\n" + "=" * 60)
    print("EDA 3 | GEOGRAPHIC DISTRIBUTION")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Geographic Crime Distribution", fontsize=14, fontweight="bold")

    # Crime count by area (sorted)
    area_counts = df["AREA NAME"].value_counts()
    area_colors = [ACCENT if i == 0 else NEUTRAL for i in range(len(area_counts))]
    axes[0].barh(area_counts.index[::-1], area_counts.values[::-1], color=area_colors[::-1])
    fmt_thousands(axes[0], axis="x")
    axes[0].set_title("Total Crimes by LAPD Patrol Area")
    axes[0].set_xlabel("Number of Crimes")
    for i, v in enumerate(area_counts.values[::-1]):
        axes[0].text(v + 20, i, f"{v:,}", va="center", fontsize=7)

    # Stacked bar: top 8 areas by crime CATEGORY composition
    top_areas = area_counts.head(8).index
    area_cat = (
        df[df["AREA NAME"].isin(top_areas)]
        .groupby(["AREA NAME", "Crime Category"])
        .size()
        .unstack(fill_value=0)
    )
    area_cat.loc[top_areas].plot(kind="barh", stacked=True, ax=axes[1], colormap="tab20")
    axes[1].set_title("Crime Composition — Top 8 Areas\n(What types drive each area's count?)")
    axes[1].set_xlabel("Number of Crimes")
    axes[1].legend(title="Crime Category", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fmt_thousands(axes[1], axis="x")
    _source_note(axes[1])

    plt.tight_layout()
    _save(fig, "03_geographic_distribution")

    print(f"  Highest-crime area: {area_counts.idxmax()} ({area_counts.max():,} crimes)")
    print(f"  Top 5 areas account for {area_counts.head(5).sum()/len(df)*100:.1f}% of all crimes")


# ── EDA 4: Victim Demographics ────────────────────────────────────────────────

def eda_victim_demographics(df: pd.DataFrame):
    """
    Q: Who are the victims?
    Critical: visualise how much we actually *don't know*, not just what we do.
    """
    print("\n" + "=" * 60)
    print("EDA 4 | VICTIM DEMOGRAPHICS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Victim Demographics (⚠ 48.5% age unknown — interpret carefully)",
                 fontsize=13, fontweight="bold", color=ACCENT)

    # Age histogram (known ages only) with median line
    valid_ages = df[(df["Vict Age"].notna()) & df["Vict Age"].between(1, 100)]["Vict Age"]
    axes[0, 0].hist(valid_ages, bins=30, color=NEUTRAL, edgecolor="white", alpha=0.85)
    med_age = valid_ages.median()
    axes[0, 0].axvline(med_age, color=ACCENT, linewidth=2, label=f"Median: {med_age:.0f} yrs")
    axes[0, 0].set_title(f"Victim Age Distribution\n(Known ages only, n={len(valid_ages):,})")
    axes[0, 0].set_xlabel("Age")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()
    fmt_thousands(axes[0, 0])

    # Age group breakdown
    if "Age Group" in df.columns:
        ag = df["Age Group"].value_counts().sort_index()
        axes[0, 1].bar(range(len(ag)), ag.values, color=NEUTRAL, edgecolor="white")
        axes[0, 1].set_xticks(range(len(ag)))
        axes[0, 1].set_xticklabels(ag.index, rotation=35, ha="right")
        axes[0, 1].set_title("Crimes by Victim Age Group")
        axes[0, 1].set_ylabel("Number of Crimes")
        fmt_thousands(axes[0, 1])

    # Sex distribution — including Unknown as its own visible slice
    sex_counts = df["Vict Sex Clean"].value_counts()
    explode = [0.05 if s == "Unknown" else 0 for s in sex_counts.index]
    axes[1, 0].pie(sex_counts.values, labels=sex_counts.index,
                   autopct="%1.1f%%", explode=explode,
                   colors=["#4C72B0", "#DD8452", "#B0B0B0"],
                   startangle=90, wedgeprops={"edgecolor": "white"})
    axes[1, 0].set_title("Victim Sex Distribution\n(Exploded = Unknown)")

    # Descent — top 8 known + Unknown
    descent_counts = df["Vict Descent Clean"].value_counts().head(9)
    bar_colors = [ACCENT if d == "Unknown" else NEUTRAL for d in descent_counts.index]
    axes[1, 1].barh(descent_counts.index[::-1], descent_counts.values[::-1], color=bar_colors[::-1])
    axes[1, 1].set_title("Victim Descent (Top 9)\nRed = Unknown — dominant category")
    axes[1, 1].set_xlabel("Number of Crimes")
    fmt_thousands(axes[1, 1], axis="x")
    _source_note(axes[1, 1])

    plt.tight_layout()
    _save(fig, "04_victim_demographics")

    print(f"  Median victim age (known): {med_age:.1f}")
    print(f"  Most common sex: {sex_counts.idxmax()} ({sex_counts.max():,})")
    print(f"  % with unknown descent: {(df['Vict Descent Clean']=='Unknown').mean()*100:.1f}%")


# ── EDA 5: Crime Type Deep-Dive ───────────────────────────────────────────────

def eda_crime_types(df: pd.DataFrame):
    """
    Q: What crime types dominate and what is their severity breakdown?
    """
    print("\n" + "=" * 60)
    print("EDA 5 | CRIME TYPE ANALYSIS")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Crime Type & Severity Breakdown", fontsize=14, fontweight="bold")

    # Crime categories sorted
    crime_cat = df["Crime Category"].value_counts()
    bar_colors = [ACCENT if i == 0 else NEUTRAL for i in range(len(crime_cat))]
    axes[0].barh(crime_cat.index[::-1], crime_cat.values[::-1], color=bar_colors[::-1])
    axes[0].set_title("Crimes by Category")
    axes[0].set_xlabel("Number of Crimes")
    fmt_thousands(axes[0], axis="x")
    for i, v in enumerate(crime_cat.values[::-1]):
        axes[0].text(v + 50, i, f"{v:,}", va="center", fontsize=7)

    # Severity distribution
    if "Severity" in df.columns:
        sev = df["Severity"].value_counts().reindex(["High", "Medium", "Low"])
        palette = {"High": ACCENT, "Medium": "#FF7F0E", "Low": NEUTRAL}
        axes[1].bar(sev.index, sev.values, color=[palette[s] for s in sev.index], edgecolor="white")
        axes[1].set_title("Crime Severity Tiers\n(High: Homicide/Sex/Robbery/Assault)")
        axes[1].set_ylabel("Number of Crimes")
        fmt_thousands(axes[1])
        for i, (lvl, v) in enumerate(zip(sev.index, sev.values)):
            axes[1].text(i, v + 100, f"{v/len(df)*100:.1f}%", ha="center", fontsize=10, fontweight="bold")
        _source_note(axes[1])

    plt.tight_layout()
    _save(fig, "05_crime_types")

    print(f"  Top crime category: {crime_cat.idxmax()} ({crime_cat.max():,} crimes, "
          f"{crime_cat.max()/len(df)*100:.1f}%)")


# ── EDA 6: HYPOTHESIS TEST — Vehicle Crime Location ──────────────────────────

def eda_test_vehicle_crime_hypothesis(df: pd.DataFrame):
    """
    HYPOTHESIS from slide 3:
    'Vehicle crimes are more frequent in parking areas than on streets.'

    We test this directly with counts and proportions.
    """
    print("\n" + "=" * 60)
    print("EDA 6 | HYPOTHESIS TEST: Vehicle Crime by Premise")
    print("=" * 60)

    if "Premise Category" not in df.columns or "Crime Category" not in df.columns:
        print("  ⚠ Missing columns — skipping hypothesis test")
        return

    vehicle_crimes = df[df["Crime Category"] == "Vehicle Crime"]
    premise_counts = vehicle_crimes["Premise Category"].value_counts()
    total_by_premise = df["Premise Category"].value_counts()

    # Rate = vehicle crimes / all crimes in that premise type
    rate = (premise_counts / total_by_premise * 100).dropna().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hypothesis: Vehicle Crimes by Location Type", fontsize=14, fontweight="bold")

    # Raw counts
    highlight = ["Parking Area", "Public Street", "Residential", "Vehicle"]
    colors = [ACCENT if p in highlight else NEUTRAL for p in premise_counts.index]
    axes[0].barh(premise_counts.index[::-1], premise_counts.values[::-1],
                 color=colors[::-1])
    axes[0].set_title("Vehicle Crime Count by Premise")
    axes[0].set_xlabel("Number of Vehicle Crimes")
    fmt_thousands(axes[0], axis="x")

    # Rate — normalised by how common each premise type is overall
    colors2 = [ACCENT if p in highlight else NEUTRAL for p in rate.index]
    axes[1].barh(rate.index[::-1], rate.values[::-1], color=colors2[::-1])
    axes[1].set_title("Vehicle Crime RATE by Premise\n(% of all crimes at that location)")
    axes[1].set_xlabel("% That Are Vehicle Crimes")
    for i, v in enumerate(rate.values[::-1]):
        axes[1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    _save(fig, "06_hypothesis_vehicle_crime_location")

    # Print verdict
    if "Parking Area" in rate.index and "Public Street" in rate.index:
        parking_rate = rate.get("Parking Area", 0)
        street_rate  = rate.get("Public Street", 0)
        verdict = "SUPPORTED ✓" if parking_rate > street_rate else "NOT SUPPORTED ✗"
        print(f"\n  HYPOTHESIS VERDICT: {verdict}")
        print(f"    Vehicle crime rate in Parking Areas: {parking_rate:.1f}%")
        print(f"    Vehicle crime rate on Public Streets: {street_rate:.1f}%")


# ── EDA 7: Reporting Patterns ─────────────────────────────────────────────────

def eda_reporting_patterns(df: pd.DataFrame):
    """
    Q: How quickly are crimes reported? Does delay vary by crime type?
    Delayed reporting can mean cold cases or domestic situations.
    """
    print("\n" + "=" * 60)
    print("EDA 7 | REPORTING PATTERNS")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Crime Reporting Delay Analysis", fontsize=14, fontweight="bold")

    # Delay distribution (capped at 60 days to show meaningful range)
    delays_capped = df["Reporting Delay (Days)"].clip(upper=60)
    axes[0].hist(delays_capped, bins=60, color=NEUTRAL, edgecolor="white", alpha=0.85)
    med = df["Reporting Delay (Days)"].median()
    axes[0].axvline(med, color=ACCENT, linewidth=2, label=f"Median: {med:.0f} days")
    axes[0].set_title("Reporting Delay Distribution\n(Capped at 60 days for readability)")
    axes[0].set_xlabel("Days Between Crime & Report")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    fmt_thousands(axes[0])

    # Median delay by crime category — some types take far longer to report
    cat_delay = (
        df.groupby("Crime Category")["Reporting Delay (Days)"]
        .median()
        .sort_values(ascending=False)
    )
    bar_colors = [ACCENT if d > 7 else NEUTRAL for d in cat_delay.values]
    axes[1].barh(cat_delay.index[::-1], cat_delay.values[::-1], color=bar_colors[::-1])
    axes[1].set_title("Median Reporting Delay by Crime Type\n(Red = delayed > 1 week — often domestic/trauma)")
    axes[1].set_xlabel("Median Days to Report")
    for i, v in enumerate(cat_delay.values[::-1]):
        axes[1].text(v + 0.1, i, f"{v:.1f}d", va="center", fontsize=8)
    _source_note(axes[1])

    plt.tight_layout()
    _save(fig, "07_reporting_patterns")

    print(f"  Median reporting delay: {med:.0f} days")
    print(f"  Same-day reports: {(df['Reporting Delay (Days)'] == 0).mean()*100:.1f}%")
    if "SuspiciousDelay" in df.columns:
        print(f"  Reports >365 days delayed: {df['SuspiciousDelay'].sum():,}")


# ── EDA 8: Demographic × Crime Cross-Tab ─────────────────────────────────────

def eda_demographic_crime_crosstab(df: pd.DataFrame):
    """
    Q: Does crime type vary by victim sex/age group?
    Shown as percentages (not raw counts) to account for group size differences.
    """
    print("\n" + "=" * 60)
    print("EDA 8 | DEMOGRAPHICS × CRIME TYPE")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Does Victim Demographics Predict Crime Type?", fontsize=14, fontweight="bold")

    # Sex × Crime Category (% within each sex — removes size bias)
    known_sex = df[df["Vict Sex Clean"].isin(["Male", "Female"])]
    sex_cross = pd.crosstab(
        known_sex["Crime Category"],
        known_sex["Vict Sex Clean"],
        normalize="columns"
    ) * 100
    sex_cross.plot(kind="barh", ax=axes[0], color=[NEUTRAL, ACCENT], edgecolor="white")
    axes[0].set_title("Crime Category by Victim Sex\n(% within each sex, unknown excluded)")
    axes[0].set_xlabel("% of Crimes for That Sex")
    axes[0].legend(title="Sex", fontsize=9)
    axes[0].set_ylabel("")

    # Age Group × Top 5 Crime Categories — heatmap
    if "Age Group" in df.columns:
        top5 = df["Crime Category"].value_counts().head(5).index
        age_cross = pd.crosstab(
            df[df["Crime Category"].isin(top5)]["Age Group"],
            df[df["Crime Category"].isin(top5)]["Crime Category"],
            normalize="index"
        ) * 100
        sns.heatmap(age_cross, ax=axes[1], cmap=PALETTE,
                    annot=True, fmt=".1f", linewidths=0.5,
                    cbar_kws={"label": "% of Age Group"})
        axes[1].set_title("Crime Type Distribution by Age Group (%)\nWhat crimes affect which age groups most?")
        axes[1].set_ylabel("Age Group")
        _source_note(axes[1])

    plt.tight_layout()
    _save(fig, "08_demographic_crime_crosstab")


# ── EDA 9: Weapon Usage Deep-Dive ────────────────────────────────────────────

def eda_weapon_analysis(df: pd.DataFrame):
    """
    Q: Where and when are weapons involved?
    Weapon presence escalates severity — important for resource planning.
    """
    print("\n" + "=" * 60)
    print("EDA 9 | WEAPON USAGE ANALYSIS")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Weapon Usage in Crimes", fontsize=14, fontweight="bold")

    # Weapon rate by crime category
    if "Has Weapon" in df.columns:
        weapon_by_cat = (
            df.groupby("Crime Category")["Has Weapon"]
            .mean()
            .sort_values(ascending=False) * 100
        )
        bar_colors = [ACCENT if v > 20 else NEUTRAL for v in weapon_by_cat.values]
        axes[0].barh(weapon_by_cat.index[::-1], weapon_by_cat.values[::-1], color=bar_colors[::-1])
        axes[0].set_title("% of Crimes Involving a Weapon\nby Crime Category")
        axes[0].set_xlabel("% With Weapon")
        for i, v in enumerate(weapon_by_cat.values[::-1]):
            axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)

    # Top weapon types (when weapon IS present)
    if "Weapon Desc" in df.columns:
        top_weapons = df[df["Has Weapon"]]["Weapon Desc"].value_counts().head(10)
        axes[1].barh(top_weapons.index[::-1], top_weapons.values[::-1], color=NEUTRAL)
        axes[1].set_title("Top 10 Weapon Types\n(Among the 5.9% of crimes with weapons)")
        axes[1].set_xlabel("Number of Crimes")
        fmt_thousands(axes[1], axis="x")
        _source_note(axes[1])

    plt.tight_layout()
    _save(fig, "09_weapon_analysis")

    if "Has Weapon" in df.columns:
        print(f"  Crimes with weapon: {df['Has Weapon'].sum():,} ({df['Has Weapon'].mean()*100:.1f}%)")


# ── EDA 10: Summary Statistics Table ─────────────────────────────────────────

def eda_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prints and returns a clean summary stats table.
    """
    print("\n" + "=" * 60)
    print("EDA 10 | SUMMARY STATISTICS")
    print("=" * 60)

    numeric_cols = [c for c in ["Vict Age", "Hour", "Reporting Delay (Days)"] if c in df.columns]
    summary = df[numeric_cols].describe().round(2)
    print(summary.to_string())

    print("\nTop 5 Crime Categories:")
    print(df["Crime Category"].value_counts().head(5).to_string())

    print("\nTop 5 LAPD Areas by Crime Count:")
    print(df["AREA NAME"].value_counts().head(5).to_string())

    return summary


# ── Pipeline Orchestrator ─────────────────────────────────────────────────────

def run_eda(cleaned_data_path: str):
    """
    Run the full EDA pipeline in one call.
    All figures saved to figures/ directory.
    """
    df = load_cleaned_data(cleaned_data_path)

    eda_data_quality(df)
    eda_temporal_trends(df)
    eda_geographic(df)
    eda_victim_demographics(df)
    eda_crime_types(df)
    eda_test_vehicle_crime_hypothesis(df)
    eda_reporting_patterns(df)
    eda_demographic_crime_crosstab(df)
    eda_weapon_analysis(df)
    eda_summary_statistics(df)

    print("\n" + "=" * 60)
    print(f"✓ EDA COMPLETE — {len(list(FIG_DIR.glob('*.png')))} figures saved to {FIG_DIR}/")
    print("=" * 60)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_eda("data/processed/crime_data_cleaned.csv")
