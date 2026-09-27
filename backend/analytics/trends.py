"""
trends.py — Member 1 Deliverable
===================================
Computes trends, period-over-period comparisons, category/region rankings,
and anomaly detections.  Produces output conforming to the shared analytics
JSON schema (trends / comparisons / anomalies / distributions sections).

Design rules
------------
- All methods accept a clean DataFrame and return plain Python dicts/lists
  that are JSON-serialisable.
- Statistical thresholds are documented and kept as named constants.
- No side effects; no global mutable state.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .kpis import (
    monthly_revenue,
    quarterly_revenue,
    period_over_period_growth,
    revenue_by_category,
    revenue_by_region,
    revenue_by_product,
)

logger = logging.getLogger(__name__)

# ── Thresholds (documented) ───────────────────────────────────────────────────
# A trend is classified as "increasing" when the linear slope of the monthly
# revenue time series is positive AND cumulative growth > TREND_MIN_GROWTH_PCT.
TREND_MIN_GROWTH_PCT: float = 5.0    # percent

# A month-over-month change is "significant" when its absolute value exceeds
# this threshold (percent).
SIGNIFICANT_CHANGE_PCT: float = 10.0

# Anomaly detection uses the IQR method on the daily revenue aggregation.
# A data point is an anomaly when it lies outside
# [Q1 - IQR_MULTIPLIER × IQR, Q3 + IQR_MULTIPLIER × IQR].
IQR_MULTIPLIER: float = 1.5

# A data point with severity "high" is one that falls outside
# IQR_HIGH_MULTIPLIER × IQR.
IQR_HIGH_MULTIPLIER: float = 3.0


# ── Trend analysis ────────────────────────────────────────────────────────────

def detect_revenue_trend(df: pd.DataFrame) -> dict:
    """Classify the overall revenue trend using linear regression on monthly totals.

    Method
    ------
    1. Aggregate revenue by month (monthly_revenue helper).
    2. Fit a linear regression on (month_index, revenue).
    3. Compute cumulative percent change from first to last month.
    4. Classify direction as:
       - "increasing"  : slope > 0 AND cumulative change > +TREND_MIN_GROWTH_PCT
       - "decreasing"  : slope < 0 AND cumulative change < -TREND_MIN_GROWTH_PCT
       - "stable"      : otherwise

    Returns
    -------
    dict
        Conforms to the ``trends`` array element in the analytics JSON schema.
    """
    monthly = monthly_revenue(df)
    if len(monthly) < 2:
        return {}

    x = np.arange(len(monthly), dtype=float)
    y = monthly["revenue"].values.astype(float)

    # Ordinary least-squares slope
    slope = float(np.polyfit(x, y, 1)[0])

    first_val = float(y[0]) if y[0] != 0 else 1.0
    last_val  = float(y[-1])
    cumulative_change_pct = ((last_val - first_val) / abs(first_val)) * 100

    if slope > 0 and cumulative_change_pct > TREND_MIN_GROWTH_PCT:
        direction = "increasing"
    elif slope < 0 and cumulative_change_pct < -TREND_MIN_GROWTH_PCT:
        direction = "decreasing"
    else:
        direction = "stable"

    significance = "high" if abs(cumulative_change_pct) > 15 else "medium"

    start_date = df["date"].min().strftime("%Y-%m-%d")
    end_date   = df["date"].max().strftime("%Y-%m-%d")

    return {
        "metric":    "Revenue",
        "dimension": "date",
        "direction": direction,
        "change":    round(cumulative_change_pct, 2),
        "unit":      "percent",
        "period":    {"start": start_date, "end": end_date},
        "significance": significance,
    }


def monthly_growth_series(df: pd.DataFrame) -> list[dict]:
    """Return month-over-month revenue growth for every month.

    Returns
    -------
    list[dict]
        Each dict has: period_key, metric_total, prev_total, growth_pct.
    """
    pop = period_over_period_growth(df, period="month", metric="sales")
    records = []
    for _, row in pop.iterrows():
        records.append({
            "period_key":   row["period_key"],
            "metric_total": round(float(row["metric_total"]), 2),
            "prev_total":   round(float(row["prev_total"]), 2) if not np.isnan(row["prev_total"]) else None,
            "growth_pct":   round(float(row["growth_pct"]), 2) if not np.isnan(row["growth_pct"]) else None,
        })
    return records


def quarterly_growth_series(df: pd.DataFrame) -> list[dict]:
    """Return quarter-over-quarter revenue growth.

    Returns
    -------
    list[dict]
        Each dict has: period_key, metric_total, prev_total, growth_pct.
    """
    pop = period_over_period_growth(df, period="quarter", metric="sales")
    records = []
    for _, row in pop.iterrows():
        records.append({
            "period_key":   row["period_key"],
            "metric_total": round(float(row["metric_total"]), 2),
            "prev_total":   round(float(row["prev_total"]), 2) if not np.isnan(row["prev_total"]) else None,
            "growth_pct":   round(float(row["growth_pct"]), 2) if not np.isnan(row["growth_pct"]) else None,
        })
    return records


# ── Comparison analysis ───────────────────────────────────────────────────────

def category_comparison(df: pd.DataFrame) -> dict:
    """Identify the highest and lowest revenue-generating categories.

    Returns
    -------
    dict
        Conforms to the ``comparisons`` array element in the analytics JSON schema.
    """
    cat_rev = revenue_by_category(df)
    return {
        "dimension": "category",
        "metric":    "Revenue",
        "highest":   {"name": str(cat_rev.index[0]),  "value": round(float(cat_rev.iloc[0]),  2)},
        "lowest":    {"name": str(cat_rev.index[-1]), "value": round(float(cat_rev.iloc[-1]), 2)},
        "all_values": [
            {"name": str(k), "value": round(float(v), 2)}
            for k, v in cat_rev.items()
        ],
    }


def region_comparison(df: pd.DataFrame) -> dict:
    """Identify the highest and lowest revenue-generating regions.

    Returns
    -------
    dict
        Conforms to the ``comparisons`` array element.
    """
    reg_rev = revenue_by_region(df)
    return {
        "dimension": "region",
        "metric":    "Revenue",
        "highest":   {"name": str(reg_rev.index[0]),  "value": round(float(reg_rev.iloc[0]),  2)},
        "lowest":    {"name": str(reg_rev.index[-1]), "value": round(float(reg_rev.iloc[-1]), 2)},
        "all_values": [
            {"name": str(k), "value": round(float(v), 2)}
            for k, v in reg_rev.items()
        ],
    }


def product_ranking(df: pd.DataFrame, top_n: int = 5) -> dict:
    """Rank products by revenue and return top-N and bottom-N.

    Parameters
    ----------
    top_n:
        How many products to include in the top and bottom lists.

    Returns
    -------
    dict
        Keys: dimension, metric, top, bottom.
    """
    prod_rev = revenue_by_product(df, top_n=None)  # all products
    return {
        "dimension": "product",
        "metric":    "Revenue",
        "top": [
            {"rank": i + 1, "name": str(k), "value": round(float(v), 2)}
            for i, (k, v) in enumerate(prod_rev.head(top_n).items())
        ],
        "bottom": [
            {"rank": len(prod_rev) - i, "name": str(k), "value": round(float(v), 2)}
            for i, (k, v) in enumerate(prod_rev.tail(top_n).iloc[::-1].items())
        ],
    }


def largest_changes(df: pd.DataFrame) -> dict:
    """Identify the largest positive and negative month-over-month changes.

    Returns
    -------
    dict
        Keys: largest_positive, largest_negative.  Each value is a dict with
        period_key and growth_pct.
    """
    pop = period_over_period_growth(df, period="month", metric="sales").dropna(subset=["growth_pct"])
    if pop.empty:
        return {"largest_positive": None, "largest_negative": None}

    pos_row = pop.loc[pop["growth_pct"].idxmax()]
    neg_row = pop.loc[pop["growth_pct"].idxmin()]

    return {
        "largest_positive": {
            "period_key": pos_row["period_key"],
            "growth_pct": round(float(pos_row["growth_pct"]), 2),
        },
        "largest_negative": {
            "period_key": neg_row["period_key"],
            "growth_pct": round(float(neg_row["growth_pct"]), 2),
        },
    }


# ── Anomaly detection ─────────────────────────────────────────────────────────

def detect_anomalies(
    df: pd.DataFrame,
    *,
    metric: str = "sales",
    dimension: str = "date",
) -> list[dict]:
    """Detect anomalous daily revenue values using the IQR method.

    Method (IQR — Interquartile Range)
    -----------------------------------
    1. Aggregate the metric by the dimension (default: daily revenue).
    2. Compute Q1, Q3, IQR = Q3 - Q1.
    3. Lower fence = Q1 - IQR_MULTIPLIER × IQR  (default 1.5)
       Upper fence = Q3 + IQR_MULTIPLIER × IQR
    4. Any value outside these fences is flagged as an anomaly.
    5. Severity is "high" when outside IQR_HIGH_MULTIPLIER × IQR (default 3.0).

    Parameters
    ----------
    metric:
        Column to aggregate (default ``"sales"``).
    dimension:
        Column to group by before aggregation (default ``"date"``).

    Returns
    -------
    list[dict]
        Each dict conforms to the ``anomalies`` array element in the JSON schema.
    """
    if dimension == "date":
        agg = df.groupby("date")[metric].sum()
    else:
        agg = df.groupby(dimension)[metric].sum()

    q1  = float(agg.quantile(0.25))
    q3  = float(agg.quantile(0.75))
    iqr = q3 - q1

    lower   = q1 - IQR_MULTIPLIER * iqr
    upper   = q3 + IQR_MULTIPLIER * iqr
    hi_upper = q3 + IQR_HIGH_MULTIPLIER * iqr
    lo_lower = q1 - IQR_HIGH_MULTIPLIER * iqr

    anomalies = []
    for label, value in agg.items():
        val = float(value)
        if val < lower or val > upper:
            if val > upper:
                a_type = "high"
                severity = "high" if val > hi_upper else "medium"
            else:
                a_type = "low"
                severity = "high" if val < lo_lower else "medium"

            obs_label = label.strftime("%Y-%m-%d") if hasattr(label, "strftime") else str(label)
            anomalies.append({
                "metric":         metric,
                "dimension":      dimension,
                "observation":    obs_label,
                "value":          round(val, 2),
                "expected_range": {
                    "lower": round(lower, 2),
                    "upper": round(upper, 2),
                },
                "type":           a_type,
                "severity":       severity,
            })

    logger.info(
        "detect_anomalies: %d anomalies found (metric=%s, dimension=%s, IQR×%.1f).",
        len(anomalies), metric, dimension, IQR_MULTIPLIER,
    )
    return anomalies


# ── Distributions ─────────────────────────────────────────────────────────────

def describe_distribution(df: pd.DataFrame, column: str = "sales") -> dict:
    """Return descriptive statistics for a numeric column.

    Returns
    -------
    dict
        Conforms to the ``distributions`` array element in the JSON schema.
    """
    s = df[column].dropna()
    return {
        "metric":  column,
        "mean":    round(float(s.mean()),   2),
        "median":  round(float(s.median()), 2),
        "minimum": round(float(s.min()),    2),
        "maximum": round(float(s.max()),    2),
        "std":     round(float(s.std()),    2),
        "q1":      round(float(s.quantile(0.25)), 2),
        "q3":      round(float(s.quantile(0.75)), 2),
    }


# ── Aggregated analytics bundle ───────────────────────────────────────────────

def compute_all_trends_and_comparisons(df: pd.DataFrame) -> dict:
    """Build the full trends / comparisons / anomalies / distributions payload.

    This is the primary output consumed by the analytics JSON builder and
    subsequently by Member 2's insight engine.

    Returns
    -------
    dict
        Keys: trends, comparisons, anomalies, distributions.
    """
    trends = []
    rev_trend = detect_revenue_trend(df)
    if rev_trend:
        trends.append(rev_trend)

    comparisons = [
        category_comparison(df),
        region_comparison(df),
        product_ranking(df),
        largest_changes(df),
    ]

    anomalies = detect_anomalies(df, metric="sales", dimension="date")

    distributions = [
        describe_distribution(df, "sales"),
        describe_distribution(df, "profit"),
        describe_distribution(df, "unit_price"),
        describe_distribution(df, "quantity"),
    ]

    logger.info(
        "compute_all_trends_and_comparisons: %d trends, %d comparisons, %d anomalies, %d distributions.",
        len(trends), len(comparisons), len(anomalies), len(distributions),
    )

    return {
        "trends":        trends,
        "comparisons":   comparisons,
        "anomalies":     anomalies[:10],  # cap at 10 for readability
        "distributions": distributions,
    }
