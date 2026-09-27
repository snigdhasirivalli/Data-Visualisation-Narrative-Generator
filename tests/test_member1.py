"""
test_member1.py — Unit tests for Member 1 deliverables
=======================================================
Tests cover preprocessing, KPI calculations, and trend/anomaly detection.
Run with:  pytest tests/test_member1.py -v
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.analytics.preprocessing import (
    validate_schema,
    inspect,
    clean,
    add_derived_fields,
    get_column_catalogue,
)
from backend.analytics.kpis import (
    total_revenue,
    total_profit,
    total_orders,
    total_quantity_sold,
    average_order_value,
    profit_margin_pct,
    period_over_period_growth,
    compute_all_kpis,
)
from backend.analytics.trends import (
    detect_revenue_trend,
    detect_anomalies,
    category_comparison,
    region_comparison,
    compute_all_trends_and_comparisons,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a minimal clean DataFrame for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    cats  = rng.choice(["Electronics", "Clothing", "Furniture"], n)
    regs  = rng.choice(["North", "South", "East"], n)
    qty   = rng.integers(1, 6, n)
    price = rng.uniform(500, 5000, n)
    sales = qty * price
    profit = sales * rng.uniform(0.1, 0.35, n)

    df = pd.DataFrame({
        "order_id":   [f"ORD{i:06d}" for i in range(1, n + 1)],
        "date":       dates,
        "product":    rng.choice(["Laptop", "T-Shirt", "Chair"], n),
        "category":   cats,
        "region":     regs,
        "quantity":   qty,
        "unit_price": price,
        "sales":      sales,
        "profit":     profit,
    })
    return add_derived_fields(df)


@pytest.fixture
def sample_df():
    return make_df(200)


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_validate_schema_passes(self, sample_df):
        """validate_schema should not raise for a compliant DataFrame."""
        validate_schema(sample_df)  # should not raise

    def test_validate_schema_fails_on_missing_column(self, sample_df):
        """validate_schema should raise ValueError when a column is missing."""
        bad_df = sample_df.drop(columns=["sales"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(bad_df)

    def test_inspect_returns_required_keys(self, sample_df):
        info = inspect(sample_df)
        assert "shape" in info
        assert "null_counts" in info
        assert "duplicate_count" in info

    def test_clean_removes_negative_sales(self):
        df = make_df(50)
        df.loc[0, "sales"] = -100
        cleaned = clean(df)
        assert (cleaned["sales"] >= 0).all()

    def test_clean_removes_duplicate_order_ids(self):
        df = make_df(50)
        # Duplicate first row
        dup = df.iloc[[0]].copy()
        df = pd.concat([df, dup], ignore_index=True)
        # raw dataframe doesn't have datetime parsed — use make_df which already has it
        # We need to feed a raw-style df to clean; reset types
        cleaned = clean(df, drop_duplicate_ids=True)
        assert cleaned.duplicated(subset=["order_id"]).sum() == 0

    def test_add_derived_fields(self, sample_df):
        assert "year" in sample_df.columns
        assert "quarter" in sample_df.columns
        assert "profit_margin" in sample_df.columns
        assert "revenue" in sample_df.columns
        assert "period_label" in sample_df.columns

    def test_profit_margin_formula(self, sample_df):
        """profit_margin = (profit / sales) * 100 for each row."""
        expected = (sample_df["profit"] / sample_df["sales"]) * 100
        pd.testing.assert_series_equal(
            sample_df["profit_margin"].round(4),
            expected.round(4),
            check_names=False,
        )

    def test_column_catalogue_completeness(self):
        cat = get_column_catalogue()
        assert "sales" in cat
        assert "profit_margin" in cat
        assert cat["profit_margin"]["unit"] == "%"


# ── KPI tests ─────────────────────────────────────────────────────────────────

class TestKPIs:
    def test_total_revenue(self, sample_df):
        assert total_revenue(sample_df) == pytest.approx(sample_df["sales"].sum())

    def test_total_profit(self, sample_df):
        assert total_profit(sample_df) == pytest.approx(sample_df["profit"].sum())

    def test_total_orders(self, sample_df):
        assert total_orders(sample_df) == len(sample_df)

    def test_total_quantity_sold(self, sample_df):
        assert total_quantity_sold(sample_df) == int(sample_df["quantity"].sum())

    def test_average_order_value(self, sample_df):
        expected = sample_df["sales"].sum() / len(sample_df)
        assert average_order_value(sample_df) == pytest.approx(expected)

    def test_profit_margin_pct(self, sample_df):
        expected = (sample_df["profit"].sum() / sample_df["sales"].sum()) * 100
        assert profit_margin_pct(sample_df) == pytest.approx(expected)

    def test_average_order_value_empty(self):
        empty_df = make_df(0)
        assert average_order_value(empty_df) == 0.0

    def test_period_over_period_growth_monthly(self, sample_df):
        result = period_over_period_growth(sample_df, period="month")
        assert "period_key" in result.columns
        assert "growth_pct" in result.columns
        # First row should have NaN growth (no previous period)
        assert pd.isna(result["growth_pct"].iloc[0])

    def test_period_over_period_growth_quarterly(self, sample_df):
        result = period_over_period_growth(sample_df, period="quarter")
        assert result.shape[0] >= 1

    def test_period_over_period_invalid_period(self, sample_df):
        with pytest.raises(ValueError):
            period_over_period_growth(sample_df, period="week")

    def test_compute_all_kpis_structure(self, sample_df):
        result = compute_all_kpis(sample_df)
        assert "metrics" in result
        names = [m["name"] for m in result["metrics"]]
        assert "Total Revenue" in names
        assert "Profit Margin" in names
        assert "Average Order Value" in names

    def test_kpi_values_non_negative(self, sample_df):
        result = compute_all_kpis(sample_df)
        for m in result["metrics"]:
            if m["name"] in ("Total Revenue", "Total Profit", "Total Orders",
                             "Total Quantity Sold", "Average Order Value"):
                assert m["value"] >= 0, f"{m['name']} should be non-negative"


# ── Trend & anomaly tests ─────────────────────────────────────────────────────

class TestTrends:
    def test_detect_revenue_trend_returns_dict(self, sample_df):
        trend = detect_revenue_trend(sample_df)
        assert isinstance(trend, dict)
        if trend:
            assert trend["direction"] in ("increasing", "decreasing", "stable")

    def test_detect_revenue_trend_keys(self, sample_df):
        trend = detect_revenue_trend(sample_df)
        if trend:
            for key in ("metric", "dimension", "direction", "change", "unit", "period"):
                assert key in trend, f"Missing key: {key}"

    def test_detect_anomalies_returns_list(self, sample_df):
        anomalies = detect_anomalies(sample_df)
        assert isinstance(anomalies, list)

    def test_detect_anomalies_structure(self, sample_df):
        anomalies = detect_anomalies(sample_df)
        for a in anomalies:
            assert "metric" in a
            assert "observation" in a
            assert "value" in a
            assert "expected_range" in a
            assert "type" in a
            assert a["type"] in ("high", "low")

    def test_category_comparison(self, sample_df):
        comp = category_comparison(sample_df)
        assert "highest" in comp
        assert "lowest" in comp
        assert comp["dimension"] == "category"

    def test_region_comparison(self, sample_df):
        comp = region_comparison(sample_df)
        assert comp["dimension"] == "region"

    def test_compute_all_trends_and_comparisons_keys(self, sample_df):
        result = compute_all_trends_and_comparisons(sample_df)
        for key in ("trends", "comparisons", "anomalies", "distributions"):
            assert key in result, f"Missing key: {key}"

    def test_distribution_stats(self, sample_df):
        result = compute_all_trends_and_comparisons(sample_df)
        for dist in result["distributions"]:
            assert dist["minimum"] <= dist["median"] <= dist["maximum"]
            assert dist["mean"] > 0


# ── Analytics JSON integration test ───────────────────────────────────────────

class TestAnalyticsJSON:
    def test_json_serialisable(self, sample_df, tmp_path):
        """The entire analytics payload must be JSON-serialisable."""
        from backend.analytics.kpis import compute_all_kpis
        from backend.analytics.trends import compute_all_trends_and_comparisons

        kpis = compute_all_kpis(sample_df)
        analytics = compute_all_trends_and_comparisons(sample_df)

        payload = {**kpis, **analytics}
        # Should not raise
        json_str = json.dumps(payload, default=str)
        parsed = json.loads(json_str)
        assert "metrics" in parsed
