"""
preprocess.py
=============
Data preprocessing pipeline for the UCI Chronic Kidney Disease dataset.

Responsibilities
----------------
1. Parse the raw ARFF file into a pandas DataFrame.
2. Standardise messy string values introduced by the source data.
3. Impute missing values (median for numerics, mode for categoricals).
4. Encode categorical columns as binary integers.
5. Save two artefacts to Data/processed/:
   - ckd_cleaned.csv   – imputed, encoded, ready for modelling
   - ckd_summary.csv   – per-column descriptive statistics

Usage
-----
    python preprocess.py

Run from the project root (BE-175-Project/).

Author: Justin Vo / Group BE 175
"""

import re
import pathlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent                 # BE-175-Project/
RAW_DIR = ROOT / "Training Data"
PROCESSED_DIR = ROOT / "Training Data"

ARFF_FILE = RAW_DIR / "chronic_kidney_disease.arff"
CLEANED_CSV = PROCESSED_DIR / "ckd_cleaned.csv"
SUMMARY_CSV = PROCESSED_DIR / "ckd_summary.csv"


# ---------------------------------------------------------------------------
# Step 1 – Parse ARFF
# ---------------------------------------------------------------------------
def parse_arff(filepath: pathlib.Path) -> pd.DataFrame:
    """
    Parse a Weka ARFF file into a pandas DataFrame.

    Handles:
    * @attribute declarations to infer column names and types
    * @data section (comma-separated rows)
    * Missing value token '?' → np.nan

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the .arff file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with column names from the ARFF header.
        Missing values are represented as np.nan.
    """
    column_names: list[str] = []
    in_data_section = False
    rows: list[list] = []

    with open(filepath, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            # Skip blank lines and comments
            if not line or line.startswith("%"):
                continue

            upper = line.upper()

            if upper.startswith("@ATTRIBUTE"):
                # Extract column name (first token after @attribute keyword)
                parts = line.split()
                # Column name may be quoted
                col = parts[1].strip("'\"")
                column_names.append(col)

            elif upper.startswith("@DATA"):
                in_data_section = True

            elif in_data_section:
                # Replace missing token with empty string so pandas reads NaN
                cleaned = line.replace("?", "")
                # Split on comma; strip whitespace from each token
                values = [v.strip() for v in cleaned.split(",")]
                # Convert empty strings back to np.nan
                values = [np.nan if v == "" else v for v in values]
                # Some rows in the UCI file have a trailing comma → 26 tokens.
                # Drop any extra trailing NaN columns beyond the expected count.
                expected = len(column_names)
                if len(values) > expected:
                    values = values[:expected]
                elif len(values) < expected:
                    values += [np.nan] * (expected - len(values))
                rows.append(values)

    df = pd.DataFrame(rows, columns=column_names)
    return df


# ---------------------------------------------------------------------------
# Step 2 – Clean raw string values
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Column rename map (raw ARFF names → names expected by the models)
# ---------------------------------------------------------------------------
COLUMN_RENAMES = {
    "wc":             "wbcc",   # white cell count → white blood cell count
    "rc":             "rbcc",   # red cell count   → red blood cell count
    "classification": "class",  # target label
}


# ---------------------------------------------------------------------------
# Columns that should hold only 'yes' / 'no'
YES_NO_COLS = ["htn", "dm", "cad", "pe", "ane"]

# Columns that should hold only 'normal' / 'abnormal'
NORM_ABNORM_COLS = ["rbc", "pc"]

# Columns that should hold only 'present' / 'notpresent'
PRESENT_COLS = ["pcc", "ba"]

# Columns that should hold only 'good' / 'poor'
APPET_COLS = ["appet"]

# Numeric columns (cast to float after cleaning)
NUMERIC_COLS = ["age", "bp", "bgr", "bu", "sc", "sod", "pot",
                "hemo", "pcv", "wbcc", "rbcc"]

# Ordinal-coded columns stored as strings in the ARFF
ORDINAL_COLS = ["sg", "al", "su"]


def _normalise_strings(series: pd.Series) -> pd.Series:
    """Strip whitespace and lowercase all string values in a Series."""
    return series.apply(
        lambda v: v.strip().lower() if isinstance(v, str) else v
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise raw string values in the DataFrame.

    Issues addressed:
    * Leading/trailing whitespace (e.g. ' yes' → 'yes')
    * Mixed case (e.g. 'Yes', 'YES' → 'yes')
    * Known typos in the UCI CKD file:
      - '\tno' → 'no'
      - 'ckd\t' → 'ckd'
      - 'notckd' spelling already correct

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from parse_arff().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with consistent string values.
    """
    df = df.copy()

    # Normalise all object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = _normalise_strings(df[col])

    # Fix tab characters sometimes present in the UCI file
    df["class"] = df["class"].str.replace(r"\t", "", regex=True)
    for col in YES_NO_COLS:
        if col in df.columns:
            df[col] = df[col].str.replace(r"\t", "", regex=True)

    # A handful of rows in the UCI file have values shifted one column to the
    # right due to an extra comma. Map these back to NaN so the imputer handles
    # them rather than silently dropping or misclassifying them.
    if "appet" in df.columns:
        df.loc[df["appet"] == "no", "appet"] = np.nan
    if "pe" in df.columns:
        df.loc[df["pe"] == "good", "pe"] = np.nan
    if "class" in df.columns:
        df.loc[~df["class"].isin(["ckd", "notckd"]), "class"] = np.nan

    # Cast numeric columns to float
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cast ordinal columns to float (they are numeric codes)
    for col in ORDINAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Step 3 – Impute missing values
# ---------------------------------------------------------------------------

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using a simple but clinically motivated strategy:

    * Numeric columns  → median (robust to outliers common in lab data)
    * Categorical cols → mode  (most frequent category)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from clean_dataframe().

    Returns
    -------
    pd.DataFrame
        DataFrame with no remaining NaN values.
    """
    df = df.copy()
    missing_before = df.isna().sum()

    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            strategy = "median"
        else:
            fill_val = df[col].mode()[0]
            strategy = "mode"

        df[col] = df[col].fillna(fill_val)
        print(f"  [{col}]  {n_missing} missing values → filled with {strategy} ({fill_val!r})")

    missing_after = df.isna().sum().sum()
    print(f"\n  Remaining NaN after imputation: {missing_after}")
    return df


# ---------------------------------------------------------------------------
# Step 4 – Encode categorical columns
# ---------------------------------------------------------------------------

# Explicit binary maps so encoding is deterministic and documented
BINARY_MAPS: dict[str, dict[str, int]] = {
    "rbc":   {"normal": 0, "abnormal": 1},
    "pc":    {"normal": 0, "abnormal": 1},
    "pcc":   {"notpresent": 0, "present": 1},
    "ba":    {"notpresent": 0, "present": 1},
    "htn":   {"no": 0, "yes": 1},
    "dm":    {"no": 0, "yes": 1},
    "cad":   {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
    "pe":    {"no": 0, "yes": 1},
    "ane":   {"no": 0, "yes": 1},
    "class": {"notckd": 0, "ckd": 1},
}


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace categorical string columns with integer codes.

    Encoding is done via explicit lookup dictionaries (BINARY_MAPS) rather
    than sklearn's LabelEncoder so that the mapping is stable, transparent,
    and the same whether the function is called on train or test data.

    Parameters
    ----------
    df : pd.DataFrame
        Imputed DataFrame from impute_missing().

    Returns
    -------
    pd.DataFrame
        Fully numeric DataFrame ready for scikit-learn estimators.
    """
    df = df.copy()
    for col, mapping in BINARY_MAPS.items():
        if col not in df.columns:
            continue
        df[col] = df[col].map(mapping)
        unmapped = df[col].isna().sum()
        if unmapped > 0:
            print(f"  WARNING: {unmapped} unmapped values in '{col}' – set to NaN")

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Step 5 – Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-column descriptive statistics for the processed dataset.

    Includes count, mean, std, min, 25th/50th/75th percentile, and max.
    Also appends a row showing the missing-value count *before* imputation
    was applied (the function re-derives this from the cleaned-but-not-yet-
    imputed data, so it must be called on the raw cleaned df).

    Parameters
    ----------
    df : pd.DataFrame
        Numeric DataFrame (post-encode).

    Returns
    -------
    pd.DataFrame
        Transposed describe() output, one row per feature.
    """
    summary = df.describe().T
    summary.index.name = "feature"
    return summary


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline end-to-end.

    Steps
    -----
    1. Parse ARFF → raw DataFrame
    2. Clean string values
    3. Impute missing values
    4. Encode categoricals → fully numeric DataFrame
    5. Save ckd_cleaned.csv and ckd_summary.csv

    Returns
    -------
    pd.DataFrame
        Final cleaned, imputed, encoded DataFrame.
    """
    print("=" * 60)
    print("CKD Data Preprocessing Pipeline")
    print("=" * 60)

    # 1. Parse
    print(f"\n[1/4] Parsing ARFF: {ARFF_FILE}")
    df_raw = parse_arff(ARFF_FILE)
    print(f"      Shape: {df_raw.shape}  ({df_raw.shape[0]} samples, {df_raw.shape[1]} features)")

    # Rename raw column names to match model expectations; drop id if present
    df_raw = df_raw.rename(columns=COLUMN_RENAMES)
    if "id" in df_raw.columns:
        df_raw = df_raw.drop(columns=["id"])

    # 2. Clean
    print("\n[2/4] Cleaning string values …")
    df_clean = clean_dataframe(df_raw)
    total_missing = df_clean.isna().sum().sum()
    print(f"      Total missing values (after cleaning): {total_missing}")
    # Per-column missing count
    missing_counts = df_clean.isna().sum()
    for col, cnt in missing_counts[missing_counts > 0].items():
        pct = 100 * cnt / len(df_clean)
        print(f"        {col:10s}: {cnt:3d} ({pct:.1f}%)")

    # 3. Impute
    print("\n[3/4] Imputing missing values …")
    df_imputed = impute_missing(df_clean)

    # 4. Encode
    print("\n[4/4] Encoding categorical columns …")
    df_final = encode_categoricals(df_imputed)
    print(f"      Final dtypes:\n{df_final.dtypes.to_string()}")

    # 5. Save
    df_final.to_csv(CLEANED_CSV, index=False)
    print(f"\n✓  Cleaned dataset saved → {CLEANED_CSV}")

    summary = compute_summary(df_final)
    summary.to_csv(SUMMARY_CSV)
    print(f"✓  Summary statistics saved → {SUMMARY_CSV}")

    # Quick sanity check on class distribution
    class_counts = df_final["class"].value_counts().rename({1: "CKD", 0: "Not CKD"})
    print(f"\nClass distribution:\n{class_counts.to_string()}")

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print("=" * 60)
    return df_final


if __name__ == "__main__":
    run_pipeline()
