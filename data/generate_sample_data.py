"""
Script to generate a sample business sales dataset (sales.csv).
Run this once to produce the data file used by the analytics pipeline.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Configuration ──────────────────────────────────────────────────────────────
N_RECORDS = 12_500
START_DATE = datetime(2025, 1, 1)
END_DATE   = datetime(2025, 12, 31)

PRODUCTS = {
    "Electronics": ["Laptop", "Smartphone", "Tablet", "Headphones", "Smartwatch"],
    "Furniture":   ["Chair", "Desk", "Bookshelf", "Sofa", "Wardrobe"],
    "Clothing":    ["T-Shirt", "Jeans", "Jacket", "Dress", "Shoes"],
    "Groceries":   ["Rice", "Cooking Oil", "Spices Pack", "Dairy Bundle", "Snacks"],
    "Sports":      ["Yoga Mat", "Dumbbell Set", "Running Shoes", "Cycle", "Badminton Kit"],
}

REGIONS = ["North", "South", "East", "West", "Central"]

# Category-level base price ranges (INR)
CATEGORY_BASE = {
    "Electronics": (8_000, 80_000),
    "Furniture":   (2_000, 30_000),
    "Clothing":    (300,   5_000),
    "Groceries":   (100,   2_000),
    "Sports":      (500,   15_000),
}

# Category-level profit margin range (fraction)
MARGIN_RANGE = {
    "Electronics": (0.08, 0.20),
    "Furniture":   (0.10, 0.25),
    "Clothing":    (0.20, 0.45),
    "Groceries":   (0.05, 0.15),
    "Sports":      (0.15, 0.35),
}

def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def build_dataset(n: int) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        category = random.choice(list(PRODUCTS.keys()))
        product  = random.choice(PRODUCTS[category])
        region   = random.choice(REGIONS)
        date     = random_date(START_DATE, END_DATE)

        lo, hi   = CATEGORY_BASE[category]
        unit_price = round(np.random.uniform(lo, hi), 2)

        quantity   = random.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[30, 25, 18, 12, 7, 4, 2, 2])[0]
        sales      = round(unit_price * quantity, 2)

        m_lo, m_hi = MARGIN_RANGE[category]
        margin     = np.random.uniform(m_lo, m_hi)
        profit     = round(sales * margin, 2)

        # Inject ~2 % anomalies (unusually high sales)
        if random.random() < 0.02:
            sales  = round(sales * random.uniform(3, 6), 2)
            profit = round(sales * margin, 2)

        rows.append({
            "order_id":   f"ORD{i:06d}",
            "date":       date.strftime("%Y-%m-%d"),
            "product":    product,
            "category":   category,
            "region":     region,
            "quantity":   quantity,
            "unit_price": unit_price,
            "sales":      sales,
            "profit":     profit,
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "sales.csv")
    df = build_dataset(N_RECORDS)
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path} ({len(df):,} records)")
    print(df.dtypes)
    print(df.head())
