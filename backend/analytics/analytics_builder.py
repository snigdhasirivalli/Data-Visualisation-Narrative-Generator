"""
analytics_builder.py — Member 1 Deliverable
=============================================
Orchestrator that combines preprocessing + KPI + trend outputs into the
shared analytics JSON contract defined in the project specification.

Usage (standalone):
    python analytics_builder.py --csv data/sales.csv --out data/analytics_output.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.analytics.preprocessing import preprocess, get_column_catalogue
from backend.analytics.kpis import compute_all_kpis, monthly_revenue, quarterly_revenue
from backend.analytics.trends import compute_all_trends_and_comparisons

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def build_analytics_json(csv_path: str | Path) -> dict:
    """Run the full Member-1 pipeline and return the analytics JSON dict.

    Parameters
    ----------
    csv_path:
        Path to the raw sales CSV file.

    Returns
    -------
    dict
        Complete analytics JSON conforming to the shared project schema.
    """
    csv_path = Path(csv_path)
    logger.info("Starting analytics pipeline for '%s'.", csv_path)

    # ── 1. Preprocess ─────────────────────────────────────────────────────────
    df = preprocess(csv_path)

    # ── 2. Dataset metadata ───────────────────────────────────────────────────
    catalogue = get_column_catalogue()
    numeric_cols      = [c for c, m in catalogue.items() if m["type"] == "numeric"]
    categorical_cols  = [c for c, m in catalogue.items() if m["type"] == "categorical"]
    temporal_cols     = [c for c, m in catalogue.items() if m["type"] == "temporal"]
    identifier_cols   = [c for c, m in catalogue.items() if m["type"] == "identifier"]

    dataset_meta = {
        "name":         csv_path.name,
        "domain":       "sales",
        "record_count": len(df),
        "date_range":   {
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end":   df["date"].max().strftime("%Y-%m-%d"),
        },
        "columns": {
            "numeric":     [c for c in numeric_cols      if c in df.columns],
            "categorical": [c for c in categorical_cols  if c in df.columns],
            "temporal":    [c for c in temporal_cols     if c in df.columns],
            "identifier":  [c for c in identifier_cols   if c in df.columns],
        },
        "column_catalogue": {
            col: {k: v for k, v in meta.items()}
            for col, meta in catalogue.items()
            if col in df.columns
        },
    }

    # ── 3. KPIs ───────────────────────────────────────────────────────────────
    kpis_payload = compute_all_kpis(df)

    # ── 4. Trends, comparisons, anomalies, distributions ─────────────────────
    analytics_payload = compute_all_trends_and_comparisons(df)

    # ── 5. Time-series data (for visualization layer / Member 2) ─────────────
    monthly  = monthly_revenue(df)
    quarterly = quarterly_revenue(df)

    visualizations = [
        {
            "title":     "Monthly Revenue",
            "type":      "line",
            "metric":    "Revenue",
            "dimension": "period",
            "data":      [
                {"period": f"{r['year']}-{str(r['month']).zfill(2)}", "value": float(r['revenue'])}
                for r in monthly.to_dict(orient="records")
            ],
            "summary":   (
                f"Revenue trended {analytics_payload['trends'][0]['direction']} "
                f"({analytics_payload['trends'][0]['change']:+.1f}%) over the reporting period."
                if analytics_payload["trends"]
                else "Monthly revenue data."
            ),
        },
        {
            "title":     "Revenue by Category",
            "type":      "bar",
            "metric":    "Revenue",
            "dimension": "category",
            "data":      [
                {"category": c["name"], "value": c["value"]}
                for c in analytics_payload["comparisons"][0].get("all_values", [])
            ] if analytics_payload["comparisons"] else [],
            "summary":   (
                f"{analytics_payload['comparisons'][0]['highest']['name']} generated the highest revenue "
                f"(INR {analytics_payload['comparisons'][0]['highest']['value']:,.0f})."
                if analytics_payload["comparisons"] else ""
            ),
        },
        {
            "title":     "Revenue by Region",
            "type":      "bar",
            "metric":    "Revenue",
            "dimension": "region",
            "data":      [
                {"region": r["name"], "value": r["value"]}
                for r in analytics_payload["comparisons"][1].get("all_values", [])
            ] if len(analytics_payload["comparisons"]) > 1 else [],
            "summary":   (
                f"{analytics_payload['comparisons'][1]['highest']['name']} led all regions in revenue."
                if len(analytics_payload["comparisons"]) > 1 else ""
            ),
        },
        {
            "title":     "Quarterly Revenue",
            "type":      "bar",
            "metric":    "Revenue",
            "dimension": "period",
            "data":      [
                {"period": f"{r['year']}-{r['quarter']}", "value": float(r['revenue'])}
                for r in quarterly.to_dict(orient="records")
            ],
            "summary":   "Quarterly revenue breakdown for the reporting period.",
        },
    ]

    # ── 6. Assemble final JSON ─────────────────────────────────────────────────
    output = {
        "dataset":       dataset_meta,
        "metrics":       kpis_payload["metrics"],
        "trends":        analytics_payload["trends"],
        "comparisons":   analytics_payload["comparisons"],
        "anomalies":     analytics_payload["anomalies"],
        "distributions": analytics_payload["distributions"],
        "visualizations": visualizations,
    }

    logger.info("Analytics JSON built successfully.")
    return output


def save_analytics_json(output: dict, out_path: str | Path) -> None:
    """Write the analytics JSON dict to a file.

    Parameters
    ----------
    output:
        Analytics JSON dict (from :func:`build_analytics_json`).
    out_path:
        Destination file path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Analytics JSON written → %s", out_path)


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build analytics JSON from a sales CSV.")
    parser.add_argument("--csv", required=True, help="Path to sales CSV file.")
    parser.add_argument("--out", default="data/analytics_output.json", help="Output JSON path.")
    args = parser.parse_args()

    result = build_analytics_json(args.csv)
    save_analytics_json(result, args.out)
    print(f"\n[OK] Analytics JSON saved to: {args.out}")
    print(f"  Records processed : {result['dataset']['record_count']:,}")
    print(f"  KPIs computed     : {len(result['metrics'])}")
    print(f"  Trends detected   : {len(result['trends'])}")
    print(f"  Anomalies flagged : {len(result['anomalies'])}")
