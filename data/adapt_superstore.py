"""
adapt_superstore.py — Adapter for the Kaggle Superstore dataset
================================================================
Converts the raw Superstore CSV column names and structure into the
format expected by the Member 1 analytics pipeline.

The Superstore dataset has columns like:
  Row ID, Order ID, Order Date, Ship Date, Ship Mode, Customer ID,
  Customer Name, Segment, Country, City, State, Postal Code, Region,
  Product ID, Category, Sub-Category, Product Name, Sales, Quantity,
  Discount, Profit

Our pipeline expects:
  order_id, date, product, category, region, quantity, unit_price,
  sales, profit

This script performs the mapping and writes a clean CSV that feeds
directly into preprocessing.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ── Column mapping ────────────────────────────────────────────────────────────
COLUMN_MAP = {
    "Order ID":      "order_id",
    "Order Date":    "date",
    "Product Name":  "product",
    "Category":      "category",
    "Region":        "region",
    "Quantity":      "quantity",
    "Sales":         "sales",
    "Profit":        "profit",
}

# Extra columns to keep for richer analytics
EXTRA_COLUMNS = {
    "Sub-Category":  "sub_category",
    "Segment":       "segment",
    "State":         "state",
    "City":          "city",
    "Ship Mode":     "ship_mode",
    "Discount":      "discount",
}


def adapt_superstore(
    raw_path: str | Path,
    out_path: str | Path,
    *,
    encoding: str = "latin-1",
) -> pd.DataFrame:
    """Read the raw Superstore CSV and write a pipeline-compatible CSV.

    Parameters
    ----------
    raw_path:
        Path to ``Sample - Superstore.csv``.
    out_path:
        Destination path for the adapted CSV (e.g. ``data/sales.csv``).
    encoding:
        File encoding for the raw CSV (Superstore uses latin-1).

    Returns
    -------
    pd.DataFrame
        The adapted DataFrame (also saved to ``out_path``).
    """
    raw_path = Path(raw_path)
    out_path = Path(out_path)

    df = pd.read_csv(raw_path, encoding=encoding)
    logger.info("Loaded raw Superstore: %d rows x %d cols", len(df), df.shape[1])

    # Rename core columns
    rename_map = {**COLUMN_MAP, **EXTRA_COLUMNS}
    df = df.rename(columns=rename_map)

    # Compute unit_price = sales / quantity (handle zero quantity)
    df["unit_price"] = df.apply(
        lambda r: round(r["sales"] / r["quantity"], 2) if r["quantity"] > 0 else 0.0,
        axis=1,
    )

    # Parse date (Superstore uses M/D/YYYY format)
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Select and order columns
    core_cols = ["order_id", "date", "product", "category", "region",
                 "quantity", "unit_price", "sales", "profit"]
    extra_cols = [c for c in EXTRA_COLUMNS.values() if c in df.columns]
    keep = core_cols + extra_cols
    df = df[[c for c in keep if c in df.columns]]

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Adapted dataset saved: %s (%d rows)", out_path, len(df))

    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    raw  = Path("data/raw/Sample - Superstore.csv")
    dest = Path("data/sales.csv")

    if not raw.exists():
        print(f"ERROR: Raw file not found at {raw}")
        sys.exit(1)

    df = adapt_superstore(raw, dest)
    print(f"\nSuperstore dataset adapted successfully!")
    print(f"  Source  : {raw}")
    print(f"  Output  : {dest}")
    print(f"  Records : {len(df):,}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Date range : {df['date'].min()} to {df['date'].max()}")
    print(f"\n  Categories : {sorted(df['category'].unique())}")
    print(f"  Regions    : {sorted(df['region'].unique())}")
    print(f"\nSample rows:")
    print(df.head(5).to_string(index=False))
