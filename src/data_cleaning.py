"""
Data Cleaning Module
Loads raw pharma sales data, cleans it, and saves a processed CSV.
"""

import pandas as pd
import numpy as np
import os


RAW_PATH = "data/raw/Pharma_data_analysis.xlsx"
PROCESSED_PATH = "data/processed/cleaned_data.csv"

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}


def load_data(path: str = RAW_PATH, sheet_name: str = "Pharma_data") -> pd.DataFrame:
    """Load raw Excel data into a DataFrame."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: strip spaces, lowercase, snake_case."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def check_data_quality(df: pd.DataFrame) -> None:
    """Print basic data quality summary."""
    print("\n--- Data Quality Summary ---")
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"\nShape: {df.shape}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps: dedupe, fix dtypes, handle missing values, derive columns."""

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Ensure numeric columns are numeric
    numeric_cols = ["quantity", "price", "sales", "latitude", "longitude", "year"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where critical numeric fields are missing
    critical_cols = [c for c in ["quantity", "price", "sales"] if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    print(f"Dropped {before - len(df)} rows with missing critical numeric fields")

    # Create month number column for sorting
    if "month" in df.columns:
        df["month_num"] = df["month"].map(MONTH_MAP)

    # Create proper date column (1st of month)
    if {"year", "month_num"}.issubset(df.columns):
        df["date"] = pd.to_datetime(
            dict(year=df["year"], month=df["month_num"], day=1),
            errors="coerce"
        )

    # Verify / recompute revenue
    if {"quantity", "price", "sales"}.issubset(df.columns):
        df["calculated_sales"] = df["quantity"] * df["price"]
        mismatch = (df["calculated_sales"] - df["sales"]).abs() > 1
        print(f"Rows where Sales != Quantity*Price (tolerance=1): {mismatch.sum()}")

    return df


def save_processed(df: pd.DataFrame, path: str = PROCESSED_PATH) -> None:
    """Save cleaned dataframe to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved cleaned data to {path} ({df.shape[0]} rows, {df.shape[1]} columns)")


def run_pipeline(raw_path: str = RAW_PATH, processed_path: str = PROCESSED_PATH,
                  sheet_name: str = "Pharma_data") -> pd.DataFrame:
    """Run full cleaning pipeline."""
    df = load_data(raw_path, sheet_name)
    df = clean_column_names(df)
    check_data_quality(df)
    df = clean_data(df)
    check_data_quality(df)
    save_processed(df, processed_path)
    return df


if __name__ == "__main__":
    run_pipeline()
