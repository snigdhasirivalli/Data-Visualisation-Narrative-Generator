# GenAI-Based Data Visualisation & Narrative Generator

> **Amrita Vishwa Vidyapeetham — Semester 7 GenAI Final Case Study**

A three-member team project that automatically transforms structured business data (KPIs, trends, comparisons, chart summaries) into accurate, coherent, context-aware natural-language business narratives using Generative AI.

---

## Architecture Overview

```
USER → React Dashboard → FastAPI → CSV Upload
                                 → Preprocessing
                                 → Analytics / KPI Engine   ← Member 1
                                 → Trend + Comparison + Anomaly Engine
                                 → Visualization + Chart Summaries  ← Member 2
                                 → Structured Insights JSON
                                 → Prompt Builder
                                 → LLM (Baseline / QLoRA fine-tuned)  ← Member 3
                                 → Narrative Validation
                                 → Business Narrative
                                 → React Dashboard / Report
```

---

## Repository Structure

```
genai-business-insights/
├── backend/
│   ├── main.py
│   ├── routes/
│   │   ├── analysis.py
│   │   ├── narrative.py
│   │   └── report.py
│   ├── analytics/            ← Member 1
│   │   ├── preprocessing.py
│   │   ├── kpis.py
│   │   ├── trends.py
│   │   └── analytics_builder.py
│   ├── insights/             ← Member 2
│   │   ├── trend_detector.py
│   │   ├── anomaly_detector.py
│   │   └── chart_summary.py
│   ├── llm/                  ← Member 3
│   │   ├── prompt_builder.py
│   │   ├── inference.py
│   │   └── validation.py
│   └── schemas/
│       └── insight_schema.py
├── frontend/                 ← Member 3
├── data/
│   ├── generate_sample_data.py
│   └── sales.csv             (generated)
├── notebooks/
│   └── member1_eda.ipynb
├── training/
│   └── dataset/
├── tests/
│   └── test_member1.py
├── requirements.txt
└── README.md
```

---

## Member 1 — Data & Analytics

**Owner:** Member 1  
**Primary module:** `backend/analytics/`

### Deliverables

| File | Purpose |
|------|---------|
| [`preprocessing.py`](backend/analytics/preprocessing.py) | Load, validate, clean, and add derived fields |
| [`kpis.py`](backend/analytics/kpis.py) | KPI engine with all headline metrics |
| [`trends.py`](backend/analytics/trends.py) | Trend, comparison, and anomaly detection |
| [`analytics_builder.py`](backend/analytics/analytics_builder.py) | Orchestrator producing the analytics JSON |
| [`data/generate_sample_data.py`](data/generate_sample_data.py) | Reproducible dataset generator |
| [`tests/test_member1.py`](tests/test_member1.py) | Unit tests |

### KPI Formulas

| KPI | Formula |
|-----|---------|
| **Total Revenue** | `Σ sales` |
| **Total Profit** | `Σ profit` |
| **Total Orders** | `COUNT(order_id)` |
| **Total Quantity Sold** | `Σ quantity` |
| **Average Order Value** | `Total Revenue / Total Orders` |
| **Profit Margin** | `(Total Profit / Total Revenue) × 100` |
| **Period-over-Period Growth** | `((current − previous) / previous) × 100` |

### Trend Classification

Revenue trend is classified using **linear regression** on monthly aggregates:
- **Increasing**: slope > 0 AND cumulative change > +5 %
- **Decreasing**: slope < 0 AND cumulative change < −5 %
- **Stable**: otherwise

### Anomaly Detection

Uses the **IQR method** on daily revenue:
- Lower fence = Q1 − 1.5 × IQR
- Upper fence = Q3 + 1.5 × IQR
- Severity **high** when outside Q1 − 3.0 × IQR or Q3 + 3.0 × IQR

### Dataset Column Catalogue

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `order_id` | identifier | — | Unique order identifier |
| `date` | temporal | YYYY-MM-DD | Order date |
| `product` | categorical | — | Product name (25 distinct) |
| `category` | categorical | — | Electronics / Furniture / Clothing / Groceries / Sports |
| `region` | categorical | — | North / South / East / West / Central |
| `quantity` | numeric | units | Units sold per order (1–8) |
| `unit_price` | numeric | INR | Price per unit |
| `sales` | numeric | INR | Total revenue = unit_price × quantity |
| `profit` | numeric | INR | Net profit for the order |
| `profit_margin` | numeric | % | (profit / sales) × 100 (derived) |
| `revenue` | numeric | INR | Alias for sales (derived) |
| `year/quarter/month` | temporal | — | Calendar fields (derived) |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the sample dataset

```bash
python data/generate_sample_data.py
```

### 3. Run the analytics pipeline

```bash
python backend/analytics/analytics_builder.py --csv data/sales.csv --out data/analytics_output.json
```

### 4. Run unit tests

```bash
pytest tests/test_member1.py -v
```

---

## Analytics JSON Schema (shared contract)

```json
{
  "dataset": { "name": "...", "domain": "sales", "record_count": 12500, "columns": {...} },
  "metrics": [
    { "name": "Total Revenue", "value": 1250000, "unit": "INR", "description": "..." }
  ],
  "trends": [
    { "metric": "Revenue", "dimension": "date", "direction": "increasing",
      "change": 12.4, "unit": "percent", "period": {"start": "...", "end": "..."}, "significance": "high" }
  ],
  "comparisons": [
    { "dimension": "category", "metric": "Revenue",
      "highest": {"name": "Electronics", "value": 450000},
      "lowest":  {"name": "Furniture",   "value": 180000} }
  ],
  "anomalies": [
    { "metric": "Revenue", "dimension": "date", "observation": "2025-02-15",
      "value": 95000, "expected_range": {"lower": 40000, "upper": 65000},
      "type": "high", "severity": "medium" }
  ],
  "distributions": [
    { "metric": "sales", "mean": 2450.5, "median": 2100, "minimum": 150, "maximum": 18500 }
  ],
  "visualizations": [
    { "title": "Monthly Revenue", "type": "line", "metric": "Revenue",
      "dimension": "date", "summary": "Revenue increased consistently..." }
  ]
}
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11+ | Analytics, backend, LLM integration |
| Data | Pandas, NumPy | Cleaning, transformation, numerical analysis |
| Statistics | SciPy / Statsmodels | Statistical tests (optional) |
| Charts | Plotly / Matplotlib | Interactive and analysis plots |
| LLM | Qwen3 family | Narrative generation |
| Fine-tuning | LoRA/QLoRA (PEFT/TRL/Unsloth) | Parameter-efficient adaptation |
| Backend | FastAPI | REST API |
| Frontend | React + Vite + Tailwind CSS | Dashboard UI |
| Version control | Git + GitHub | Collaboration |

---

## Integration Sequence

1. **Member 1** produces valid analytics JSON.
2. **Member 2** consumes it and produces valid structured insights JSON.
3. **Member 3** consumes that JSON and generates narrative text.
4. **FastAPI** exposes the complete flow.
5. **React** calls the API and displays the results.

---

## Team

| Member | Module |
|--------|--------|
| Member 1 | Data & Analytics (preprocessing, KPIs, trends) |
| Member 2 | Visualization & Insight Engine (charts, anomaly rules, structured JSON) |
| Member 3 | GenAI + Backend + Frontend (LLM, QLoRA, FastAPI, React) |
