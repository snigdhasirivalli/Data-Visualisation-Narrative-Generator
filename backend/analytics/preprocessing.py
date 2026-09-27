"""
preprocessing.py — Member 1 Deliverable
========================================
Loads, validates, cleans, and transforms the sales CSV into a clean DataFrame
ready for the KPI and analytics engines.

All decisions are documented inline so the pipeline is reproducible.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS: list[str] = [
    "order_id", "date", "product", "category",
    "region", "quantity", "unit_price", "sales", "profit",
]

NUMERIC_COLUMNS:      list[str] = ["quantity", "unit_price", "sales", "profit"]
CATEGORICAL_COLUMNS:  list[str] = ["product", "category", "region"]
TEMPORAL_COLUMNS:     list[str] = ["date"]
IDENTIFIER_COLUMNS:   list[str] = ["order_id"]


# ── Public API ────────────────────────────────────────────────────────────────

def load_csv(filepath: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with basic type coercion.

    Parameters
    ----------
    filepath:
        Path to the sales CSV file.

    Returns
    -------
    pd.DataFrame
        Raw (unvalidated) DataFrame.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath, dtype={"order_id": str})
    logger.info("Loaded %d rows × %d cols from '%s'", len(df), df.shape[1], filepath)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is missing.

    Parameters
    ----------
    df:
        Raw DataFrame to validate.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed.")


def inspect(df: pd.DataFrame) -> dict:
    """Return a summary dict for EDA / logging purposes.

    Parameters
    ----------
    df:
        DataFrame to inspect (typically the raw loaded frame).

    Returns
    -------
    dict
        Keys: shape, dtypes, null_counts, duplicate_count, categorical_uniques.
    """
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated(subset=IDENTIFIER_COLUMNS).sum()),
        "categorical_uniques": {
            col: df[col].nunique() for col in CATEGORICAL_COLUMNS if col in df.columns
        },
    }


def clean(df: pd.DataFrame, *, drop_duplicate_ids: bool = True) -> pd.DataFrame:
    """Apply documented cleaning rules to produce a clean DataFrame.

    Rules applied (all documented):
    1. Parse date column to datetime; drop rows with unparseable dates.
    2. Coerce numeric columns to float; rows that fail coercion are dropped.
    3. Drop rows where any numeric KPI column is negative (data integrity check).
    4. Strip whitespace from categorical columns and normalise case to title-case.
    5. Remove duplicate order_ids only when ``drop_duplicate_ids=True``
       (default). First occurrence is kept; decision is logged.
    6. Reset index after all removals.

    Parameters
    ----------
    df:
        Raw DataFrame (output of :func:`load_csv`).
    drop_duplicate_ids:
        If True (default), keep only the first row per order_id.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()
    initial_len = len(df)

    # 1. Parse dates ─────────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates:
        logger.warning("Dropping %d rows with unparseable dates.", bad_dates)
    df = df.dropna(subset=["date"])

    # 2. Coerce numerics ─────────────────────────────────────────────────────
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_numeric = df[NUMERIC_COLUMNS].isnull().any(axis=1).sum()
    if bad_numeric:
        logger.warning("Dropping %d rows with non-numeric values in %s.", bad_numeric, NUMERIC_COLUMNS)
    df = df.dropna(subset=NUMERIC_COLUMNS)

    # 3. Drop negative values for non-profit columns ──────────────────────────
    #    Profit CAN be negative (real losses) — so we only check sales, qty, price.
    neg_mask = (df[["sales", "quantity", "unit_price"]] < 0).any(axis=1)
    if neg_mask.sum():
        logger.warning("Dropping %d rows with negative sales/quantity/unit_price values.", neg_mask.sum())
    df = df[~neg_mask]

    # 4. Normalise categoricals ──────────────────────────────────────────────
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # 5. Handle duplicates ───────────────────────────────────────────────────
    dup_count = df.duplicated(subset=IDENTIFIER_COLUMNS).sum()
    if dup_count:
        if drop_duplicate_ids:
            logger.info(
                "Removing %d duplicate order_id rows (keeping first occurrence).", dup_count
            )
            df = df.drop_duplicates(subset=IDENTIFIER_COLUMNS, keep="first")
        else:
            logger.info(
                "Found %d duplicate order_ids; keeping all (drop_duplicate_ids=False).", dup_count
            )

    # 6. Reset index ─────────────────────────────────────────────────────────
    df = df.sort_values("date").reset_index(drop=True)
    removed = initial_len - len(df)
    logger.info(
        "Cleaning complete: %d → %d rows (%d removed).", initial_len, len(df), removed
    )
    return df


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and financial derived columns.

    Derived columns added:
    - ``year``         : Calendar year (int).
    - ``quarter``      : Calendar quarter Q1–Q4 (string, e.g. "Q1").
    - ``month``        : Month number 1–12 (int).
    - ``month_name``   : Abbreviated month name (e.g. "Jan").
    - ``week``         : ISO week number (int).
    - ``profit_margin``: profit / sales, NaN when sales == 0.
    - ``revenue``      : Alias for ``sales`` to match JSON schema terminology.
    - ``period_label`` : "YYYY-Qn" string for period-over-period grouping.

    Parameters
    ----------
    df:
        Clean DataFrame (output of :func:`clean`).

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns.
    """
    df = df.copy()

    # Calendar fields
    df["year"]       = df["date"].dt.year
    df["quarter"]    = "Q" + df["date"].dt.quarter.astype(str)
    df["month"]      = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["week"]       = df["date"].dt.isocalendar().week.astype(int)

    # Financial derived
    df["profit_margin"] = np.where(
        df["sales"] != 0,
        (df["profit"] / df["sales"]) * 100,
        np.nan,
    )

    # Revenue alias (keeps column naming consistent with JSON schema)
    df["revenue"] = df["sales"]

    # Period label for PoP grouping
    df["period_label"] = df["year"].astype(str) + "-" + df["quarter"]

    logger.info("Derived fields added: year, quarter, month, month_name, week, profit_margin, revenue, period_label")
    return df


def preprocess(
    filepath: str | Path,
    *,
    drop_duplicate_ids: bool = False,
) -> pd.DataFrame:
    """End-to-end preprocessing pipeline.

    Convenience wrapper that calls :func:`load_csv`, :func:`validate_schema`,
    :func:`clean`, and :func:`add_derived_fields` in order.

    Parameters
    ----------
    filepath:
        Path to the raw sales CSV.
    drop_duplicate_ids:
        Passed through to :func:`clean`.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed DataFrame ready for KPI and analytics engines.
    """
    df = load_csv(filepath)
    validate_schema(df)
    summary = inspect(df)
    logger.info("Raw dataset summary: %s", summary)
    df = clean(df, drop_duplicate_ids=drop_duplicate_ids)
    df = add_derived_fields(df)
    logger.info("Preprocessing complete. Final shape: %s", df.shape)
    return df


def get_column_catalogue() -> dict:
    """Return documented column metadata for every column in the clean dataset.

    Returns
    -------
    dict
        Maps column name → {type, meaning, unit, expected_range}.
    """
    return {
        "order_id":      {"type": "identifier", "meaning": "Unique order identifier", "unit": None, "expected_range": "ORD000001 – ORD999999"},
        "date":          {"type": "temporal",   "meaning": "Date the order was placed", "unit": "YYYY-MM-DD", "expected_range": "2025-01-01 – 2025-12-31"},
        "product":       {"type": "categorical","meaning": "Product name", "unit": None, "expected_range": "25 distinct products"},
        "category":      {"type": "categorical","meaning": "High-level product category", "unit": None, "expected_range": "Electronics | Furniture | Clothing | Groceries | Sports"},
        "region":        {"type": "categorical","meaning": "Sales region", "unit": None, "expected_range": "North | South | East | West | Central"},
        "quantity":      {"type": "numeric",    "meaning": "Units sold per order", "unit": "units", "expected_range": "1 – 8"},
        "unit_price":    {"type": "numeric",    "meaning": "Price per unit before discount", "unit": "INR", "expected_range": "100 – 80 000"},
        "sales":         {"type": "numeric",    "meaning": "Total revenue for the order (unit_price × quantity)", "unit": "INR", "expected_range": "> 0"},
        "profit":        {"type": "numeric",    "meaning": "Net profit for the order", "unit": "INR", "expected_range": "> 0"},
        # Derived
        "year":          {"type": "numeric",    "meaning": "Calendar year", "unit": None, "expected_range": "2025"},
        "quarter":       {"type": "categorical","meaning": "Calendar quarter", "unit": None, "expected_range": "Q1 – Q4"},
        "month":         {"type": "numeric",    "meaning": "Month number", "unit": None, "expected_range": "1 – 12"},
        "month_name":    {"type": "categorical","meaning": "Abbreviated month name", "unit": None, "expected_range": "Jan – Dec"},
        "week":          {"type": "numeric",    "meaning": "ISO week number", "unit": None, "expected_range": "1 – 53"},
        "profit_margin": {"type": "numeric",    "meaning": "Profit as a percentage of sales", "unit": "%", "expected_range": "5 – 45"},
        "revenue":       {"type": "numeric",    "meaning": "Alias for sales (consistent with JSON schema)", "unit": "INR", "expected_range": "> 0"},
        "period_label":  {"type": "categorical","meaning": "Year-Quarter label for period-over-period grouping", "unit": None, "expected_range": "e.g. 2025-Q1"},
    }
