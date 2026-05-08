# LSTM-Based Stock Price Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A deep learning pipeline forecasting **Apple Inc. (AAPL)** daily closing prices using Long Short-Term Memory (LSTM) neural networks. The project benchmarks a single-layer baseline against a 5-layer stacked architecture to measure the impact of model depth on temporal prediction accuracy.

---

## Results

| Model | Test MAE (USD) | Test MSE (Scaled) | Test MAE (Scaled) |
|---|---|---|---|
| Baseline LSTM | $24.45 | 0.0245 | 0.1420 |
| **Enhanced LSTM (5-layer)** | **$18.44** | **0.0133** | **0.1071** |

The stacked architecture reduced Test MAE by **~25%**, demonstrating stronger long-range temporal pattern recognition.

---

## Architecture

### Baseline LSTM
- 1 LSTM layer · Hidden size: 64
- Fully connected output layer

### Enhanced LSTM
- 5 stacked LSTM layers · Hidden size: 128
- Dropout: 0.2 between layers
- Fully connected output layer

---

## Dataset

| Property | Value |
|---|---|
| Source | Yahoo Finance via `yfinance` |
| Ticker | AAPL (Apple Inc.) |
| Period | 2010 – 2026 |
| Total rows | ~4,060 trading days |

---

## Pipeline

1. **Ingestion** — Pull AAPL OHLCV data via `yfinance`
2. **Preprocessing** — MinMax normalization, missing value checks
3. **Sequence construction** — 30-day sliding window → next-day close prediction
4. **Chronological split** — Train 80% / Validation 10% / Test 10%
5. **Training** — Both architectures trained under identical conditions
6. **Evaluation** — MAE / MSE comparison on held-out test set

---

## Tech Stack

`Python` · `PyTorch` · `Pandas` · `NumPy` · `Matplotlib` · `Scikit-learn` · `yfinance`

---

## Setup

```bash
pip install torch pandas numpy matplotlib scikit-learn yfinance
```

Open `STAT656_Group_7_Code.ipynb` in Jupyter or Google Colab and run all cells.

---

## Files

| File | Description |
|---|---|
| `STAT656_Group_7_Code.ipynb` | Full pipeline: ingestion → preprocessing → training → evaluation |
| `STAT656_Group_7_Final_Report.pdf` | Comprehensive project report with methodology and results |

---

## Course

**STAT 656 — Applied Analytics** · Texas A&M University · Spring 2026

---

## Author

**Arvinder Mundra** · [Portfolio](https://arvindermundraa.github.io/ArvinderMundra/) · [LinkedIn](https://www.linkedin.com/in/arvinder-mundraa) · [GitHub](https://github.com/arvindermundra1208)
