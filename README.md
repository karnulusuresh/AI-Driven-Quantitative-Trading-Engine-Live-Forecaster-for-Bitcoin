# ₿ Bitcoin Dual-Goal Quant Trading Engine & Live Forecaster

An end-to-end quantitative financial analytics platform that combines machine learning and deep learning to model, predict, and backtest BTC-USD market cycles. The platform handles extreme cryptocurrency market volatility by isolating historical anomalies, validating data stationarity, and generating actionable, risk-managed forward signals via an interactive Streamlit dashboard.

---

## 📌 Project Overview & Goals

Financial time-series data like Bitcoin is highly volatile, non-stationary, and prone to extreme black-swan outliers. This project implements a **Dual-Goal Architecture** that approaches the market from two distinct computational angles to maximize predictive alpha:

1. **Goal 1: Directional Classification (XGBoost Classifier)**  
   Predicts whether tomorrow's market trend will close **UP (1)** or **DOWN/FLAT (0)**. Optimized to minimize false buy signals by tuning for high class precision and accounting for systemic class imbalances.
2. **Goal 2: Volatility Magnitude Regression (LSTM Neural Network)**  
   Utilizes a Long Short-Term Memory sequence layer to retain temporal lookback context. It estimates the **exact percentage change magnitude** of tomorrow's continuous log returns, filtering out daily market noise.

---

## 🖥️ Streamlit UI Breakdown & Interpretation

When the web application is launched, it renders an interactive dashboard divided into two core operational areas:

### 1. Live Market Signals (Tomorrow's Forecast)
This top row acts as an immediate, real-time trading decision matrix for the upcoming market session:
*   **XGBoost Direction Metric:** Displays a clear trend color code: **🟢 UP TREND** or **🔴 DOWN TREND**. It also reveals the model's exact *Confidence Percentage* (e.g., Confidence: 53.4%), derived directly from the model's internal class probability distributions.
*   **LSTM Magnitude Metric:** Outlines the forecasted continuous percentage change (e.g., `+1.85%` or `-0.42%`). It automatically processes this value against live spot market rates to calculate a concrete target price in US dollars (e.g., `Target: $84,250.00`).
*   **Ensemble Trade Action:** A custom risk-mitigation consensus engine. It outputs a bright green **🔥 STRONG BUY** alert *only* if the systems structurally align (XGBoost outputs a definitive Up-trend and the LSTM confirms a positive return > 0.5%). If the models conflict or indicate structural weakness, it shifts to a defensive, capital-preservation state: **🛡️ SIT IN CASH**.

### 2. Out-of-Sample Historical Strategy Performance
This bottom section evaluates historical validity, proving to the user whether the integrated AI system can systematically beat the market:
*   **Benchmark Bitcoin Return vs. Ensemble AI Strategy Return:** Displays side-by-side, real-time ROI metrics. It automatically computes **Alpha**—the exact percentage by which your machine learning model outperformed a passive market exposure.
*   **Interactive Equity Curve (Plotly Chart):** A dynamic, zoomable time-series line graph charting the performance of a simulated **\$10,000 portfolio** initialized over the historical out-of-sample test window. It maps out two distinct curves (`Buy & Hold BTC` vs. `Ensemble AI Strategy`), allowing users to visually inspect the exact days the model successfully dodged major historical crypto crashes by moving to cash.

---

## 🚀 Core Features & Technical Highlights

* **Advanced Outlier & Stationarity Handling:** Employs `RobustScaler` (Median and IQR scaling) to protect statistical features from being corrupted by historical crypto crashes. Validates data integrity using the **Augmented Dickey-Fuller (ADF) Test** to ensure inputs are perfectly stationary log returns.
* **Engineered Multi-Feature Lags:** Translates raw daily price and volume data into structural features, including rolling 3-day momentum trends and 7-day historical price volatility indexes.
* **Live Automated Inference Pipeline:** Automatically fetches live spot exchange data directly via the Yahoo Finance API, transforms the arrays seamlessly, and runs model forward passes on the fly.

---

## 🛠️ Technology Stack & Frameworks

* **User Interface:** Streamlit, Plotly (Interactive Charting Engine)
* **Machine Learning Model:** XGBoost
* **Deep Learning Layer:** TensorFlow / Keras (Sequential LSTM Architectures)
* **Statistical Tools:** Scikit-Learn (Robust Scaler Matrix), Statsmodels (ADF Test)
* **Data Pipelines:** Pandas, NumPy, yfinance (Yahoo Finance API Engine)
* **Runtime Infrastructure:** Python 3.13 (Models preserved via `pickle` binary serialization)

---

## 📁 Repository Directory Blueprint

```text
btc-prediction-app/
│
├── app.py                # Main Streamlit UI dashboard and backtesting script
├── requirements.txt      # Automated server dependencies mapping file
├── README.md             # Project roadmap and architecture details
└── models/
    ├── scaler.pkl        # Serialized RobustScaler file matching training distributions
    ├── classifier.pkl    # Pre-trained XGBoost Directional Classifier binary (.pkl format)
    └── regressor_lstm.h5 # Saved Keras/TensorFlow LSTM sequence weights matrix
```

---

## 🔧 Installation & Local Setup

Follow these steps to run the interactive dashboard framework locally on your terminal instance:

### 1. Clone the Project Repository
```bash
git clone github.com
cd btc-prediction-app
```

### 2. Configure Dependencies
Ensure your current directory houses the `requirements.txt` asset file, then execute:
```bash
pip install -r requirements.txt
```

### 3. Initialize the Models
Ensure your exported files (`scaler.pkl`, `classifier.pkl`, and `regressor_lstm.h5`) are located inside the `models/` directory exactly as shown in the directory blueprint above.

### 4. Deploy the Local Server Instance
```bash
streamlit run app.py
```
Your primary default web browser will automatically boot open to `http://localhost:8501`, rendering your operational quantitative interface.

---

## 📈 Production Deployment Info

This application is ready to host on cloud infrastructure.
* **Streamlit Community Cloud:** Connect this repository directly via GitHub to launch on a free cloud hosting container.
* **Hugging Face Spaces:** Create a clean space profile using the Streamlit SDK runtime, commit your file layers, and host your models directly.

*Disclaimer: This tool is built entirely for quantitative educational modeling purposes and does not constitute formal financial trading advice.*
