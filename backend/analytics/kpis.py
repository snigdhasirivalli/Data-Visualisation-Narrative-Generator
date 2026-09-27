"""
kpis.py — Member 1 Deliverable
================================
KPI engine: computes headline business metrics from a clean DataFrame.

Design rules
------------
- Every KPI function accepts a clean DataFrame and returns a raw float/int.
- Rounding is deferred to presentation time (caller decides precision).
- Functions are pure — no side-effects, no global state.
- Each function includes a docstring with the exact formula used.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Primitive KPI functions ───────────────────────────────────────────────────

def total_revenue(df: pd.DataFrame) -> float:
    """Sum of the ``sales`` column across all rows.

    Formula: Σ sales
    """
    return float(df["sales"].sum())


def total_profit(df: pd.DataFrame) -> float:
    """Sum of the ``profit`` column across all rows.

    Formula: Σ profit
    """
    return float(df["profit"].sum())


def total_orders(df: pd.DataFrame) -> int:
    """Count of unique order records (rows in the clean DataFrame).

    Formula: COUNT(order_id)
    """
    return int(len(df))


def total_quantity_sold(df: pd.DataFrame) -> int:
    """Sum of the ``quantity`` column across all rows.

    Formula: Σ quantity
    """
    return int(df["quantity"].sum())


def average_order_value(df: pd.DataFrame) -> float:
    """Mean revenue per order.

    Formula: Σ sales / COUNT(order_id)
    Returns 0.0 when there are no orders.
    """
    n = len(df)
    return float(df["sales"].sum() / n) if n > 0 else 0.0


def profit_margin_pct(df: pd.DataFrame) -> float:
    """Overall profit as a percentage of total revenue.

    Formula: (Σ profit / Σ sales) × 100
    Returns 0.0 when total revenue is 0.
    """
    rev = df["sales"].sum()
    return float((df["profit"].sum() / rev) * 100) if rev != 0 else 0.0


def revenue_by_category(df: pd.DataFrame) -> pd.Series:
    """Total revenue grouped by product category.

    Returns
    -------
    pd.Series
        Index = category name, values = total revenue (float).
    """
    return df.groupby("category")["sales"].sum().sort_values(ascending=False)


def revenue_by_region(df: pd.DataFrame) -> pd.Series:
    """Total revenue grouped by region.

    Returns
    -------
    pd.Series
        Index = region name, values = total revenue (float).
    """
    return df.groupby("region")["sales"].sum().sort_values(ascending=False)


def revenue_by_product(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Total revenue for the top-N products.

    Parameters
    ----------
    top_n:
        Number of top products to return (default 10).

    Returns
    -------
    pd.Series
        Index = product name, values = total revenue (float).
    """
    return (
        df.groupby("product")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )


def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue aggregated by year-month.

    Returns
    -------
    pd.DataFrame
        Columns: year, month, month_name, revenue.
    """
    grp = (
        df.groupby(["year", "month", "month_name"])["sales"]
        .sum()
        .reset_index()
        .rename(columns={"sales": "revenue"})
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )
    return grp


def quarterly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue aggregated by year-quarter.

    Returns
    -------
    pd.DataFrame
        Columns: year, quarter, revenue.
    """
    return (
        df.groupby(["year", "quarter"])["sales"]
        .sum()
        .reset_index()
        .rename(columns={"sales": "revenue"})
        .sort_values(["year", "quarter"])
        .reset_index(drop=True)
    )


# ── Period-over-period growth ──────────────────────────────────────────────────

def period_over_period_growth(
    df: pd.DataFrame,
    *,
    period: str = "month",
    metric: str = "sales",
) -> pd.DataFrame:
    """Calculate period-over-period percentage growth.

    Parameters
    ----------
    period:
        Aggregation granularity — ``"month"`` or ``"quarter"``.
    metric:
        Column to aggregate (default ``"sales"``).

    Returns
    -------
    pd.DataFrame
        Columns: period_key, metric_total, prev_total, growth_pct.
        ``growth_pct`` is NaN for the first period (no previous period).

    Formula
    -------
    growth_pct = ((current - previous) / previous) × 100
    """
    if period == "month":
        grp = (
            df.groupby(["year", "month"])[metric]
            .sum()
            .reset_index()
            .rename(columns={metric: "metric_total"})
            .sort_values(["year", "month"])
            .reset_index(drop=True)
        )
        grp["period_key"] = (
            grp["year"].astype(str) + "-"
            + grp["month"].astype(str).str.zfill(2)
        )
    elif period == "quarter":
        grp = (
            df.groupby(["year", "quarter"])[metric]
            .sum()
            .reset_index()
            .rename(columns={metric: "metric_total"})
            .sort_values(["year", "quarter"])
            .reset_index(drop=True)
        )
        grp["period_key"] = grp["year"].astype(str) + "-" + grp["quarter"]
    else:
        raise ValueError(f"Unsupported period '{period}'. Choose 'month' or 'quarter'.")

    grp["prev_total"]  = grp["metric_total"].shift(1)
    grp["growth_pct"]  = np.where(
        grp["prev_total"] != 0,
        ((grp["metric_total"] - grp["prev_total"]) / grp["prev_total"]) * 100,
        np.nan,
    )
    return grp[["period_key", "metric_total", "prev_total", "growth_pct"]]


# ── Aggregated KPI bundle ─────────────────────────────────────────────────────

def compute_all_kpis(df: pd.DataFrame) -> dict:
    """Compute all headline KPIs and return them as a dictionary.

    This is the primary output consumed by the analytics JSON builder.

    Returns
    -------
    dict
        Keys match the ``metrics`` section of the shared analytics JSON schema:
        name, value, unit, description.
    """
    rev       = total_revenue(df)
    prof      = total_profit(df)
    orders    = total_orders(df)
    qty       = total_quantity_sold(df)
    aov       = average_order_value(df)
    pm        = profit_margin_pct(df)

    # Best PoP growth (monthly)
    pop = period_over_period_growth(df, period="month", metric="sales")
    pop_valid = pop.dropna(subset=["growth_pct"])
    best_growth_pct = float(pop_valid["growth_pct"].max()) if not pop_valid.empty else 0.0
    worst_growth_pct = float(pop_valid["growth_pct"].min()) if not pop_valid.empty else 0.0

    metrics = [
        {
            "name":        "Total Revenue",
            "value":       rev,
            "unit":        "INR",
            "description": "Total revenue (sum of sales) across all orders and periods.",
        },
        {
            "name":        "Total Profit",
            "value":       prof,
            "unit":        "INR",
            "description": "Total profit across all orders.",
        },
        {
            "name":        "Total Orders",
            "value":       orders,
            "unit":        "orders",
            "description": "Count of individual order records in the dataset.",
        },
        {
            "name":        "Total Quantity Sold",
            "value":       qty,
            "unit":        "units",
            "description": "Sum of quantity sold across all orders.",
        },
        {
            "name":        "Average Order Value",
            "value":       aov,
            "unit":        "INR",
            "description": "Mean revenue per order = Total Revenue / Total Orders.",
        },
        {
            "name":        "Profit Margin",
            "value":       pm,
            "unit":        "percent",
            "description": "Overall profit as a percentage of revenue = (Total Profit / Total Revenue) × 100.",
        },
        {
            "name":        "Best Monthly Revenue Growth",
            "value":       best_growth_pct,
            "unit":        "percent",
            "description": "Highest month-over-month revenue growth percentage observed.",
        },
        {
            "name":        "Worst Monthly Revenue Growth",
            "value":       worst_growth_pct,
            "unit":        "percent",
            "description": "Lowest (most negative) month-over-month revenue growth percentage observed.",
        },
    ]

    logger.info("compute_all_kpis: %d KPIs computed.", len(metrics))
    return {"metrics": metrics}
