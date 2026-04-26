# LSTM-Based Stock Price Prediction 📈

This project predicts the daily closing price of Apple Inc. (AAPL) stock using Long Short-Term Memory (LSTM) neural networks.

## Project Overview

The objective is to use historical stock price data from the last **30 trading days** to predict the **next day's closing price**.

Two deep learning models were implemented:

* **Baseline LSTM**
* **Enhanced LSTM (5-layer stacked LSTM with Dropout)**

## Dataset

* Source: Yahoo Finance (`yfinance`)
* Stock: AAPL (Apple Inc.)
* Period: 2010 – 2026
* Total rows: ~4,060 trading days

## Preprocessing

* Missing value check
* MinMax normalization
* Sliding window creation (30-day input sequence)
* Chronological split:

  * Train: 80%
  * Validation: 10%
  * Test: 10%

## Models Used

### Baseline LSTM

* Single LSTM layer
* Hidden size: 64

### Enhanced LSTM

* 5 stacked LSTM layers
* Hidden size: 128
* Dropout: 0.2

## Results

| Model         | Test MAE (USD) | Test MSE (Scaled) | Test MAE (Scaled) |
| ------------- | -------------- | ------------------| ----------------- |
| Baseline LSTM | 24.45          | 0.0245            | 0.1420            |
| Enhanced LSTM | 18.44          | 0.0133            | 0.1071            |

The enhanced model achieved better performance by learning deeper temporal patterns.

## Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* yfinance

## Run the Project

1. Install dependencies

```bash
pip install torch pandas numpy matplotlib scikit-learn yfinance
```

2. Open the notebook

```bash
jupyter notebook
```

3. Run all cells in sequence

## Future Improvements

* Add volume and sentiment features
* Use Transformer models
* Multi-step forecasting

## Team Members

* Aditya Rao Ghodke
* Arvinder Singh Mundra
* Dharma Pokhrel
