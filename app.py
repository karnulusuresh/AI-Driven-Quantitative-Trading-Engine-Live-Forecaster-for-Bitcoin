import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import plotly.graph_objects as go
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

st.set_page_config(page_title="BTC-USD Trading Engine", layout="wide")
st.title("₿ Bitcoin Quant Trading Engine & Live Forecaster")

# 1. Pipeline Artifact Loader
@st.cache_resource
def load_production_artifacts():
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    reg = load_model('models/regressor_lstm.h5')
    return scaler, clf, reg

try:
    scaler, classifier, lstm_regressor = load_production_artifacts()
except FileNotFoundError:
    st.error("⚠️ Model files missing. Please ensure your saved weights exist in the 'models/' folder structure.")
    st.stop()

# 2. Optimized Live Data Pipeline (FIXED FOR MULTI-INDEX)
@st.cache_data(ttl=3600)
def build_market_dataset():
    # Force auto_adjust and explicitly prevent multi-level indexes
    data = yf.download("BTC-USD", period="3y", interval="1d", auto_adjust=True)
    
    # Safety Check: Did yfinance return any data?
    if data.empty:
        return pd.DataFrame()
        
    # Flatten columns if yfinance returned a multi-index anyway
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    
    # Ensure standard casing for columns
    data.columns = [c.capitalize() for c in data.columns]
    
    if 'Close' not in data.columns or 'Volume' not in data.columns:
        return pd.DataFrame()

    df = data[['Close', 'Volume']].copy()
    
    # Structural calculations
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    for lag in [1, 2, 3]:
        df[f'Log_Returns_Lag_{lag}'] = df['Log_Returns'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
    df['Volatility_7d'] = df['Log_Returns'].rolling(window=7).std()
    
    df['Target_Classification'] = (df['Log_Returns'].shift(-1) > 0).astype(int)
    return df.dropna().copy()

df_market = build_market_dataset()

# 🛑 SAFETY CHECK 1: Global App Halt if Data Pipeline Fails
if df_market.empty or len(df_market) < 15:
    st.error("🚨 Failed to fetch market data from API or dataset is too small to calculate rolling features. Please refresh.")
    st.stop()

# 3. Dynamic Strategy Backtester Matrix
@st.cache_data
def run_strategy_backtest(_df, split_percent=0.8):
    features = [
        'Log_Returns', 'Log_Returns_Lag_2', 'Volume_Lag_1', 
        'Volatility_7d', 'Volume_Lag_2', 'Log_Returns_Lag_3', 
        'Log_Returns_Lag_1', 'Volume_Lag_3'
    ]
    
    X = _df[features]
    y = _df['Target_Classification'].values
    
    split_idx = int(len(X) * split_percent)
    
    # Process test data
    X_test = X.iloc[split_idx:]
    
    # 🛑 SAFETY CHECK 2: Prevent RobustScaler from crashing on empty test split
    if X_test.empty or X_test.shape[0] == 0:
        return pd.DataFrame()
        
    test_dates = _df.index[split_idx:]
    actual_returns = _df['Log_Returns'].iloc[split_idx:].values
    
    X_test_scaled = scaler.transform(X_test)
    y_pred_class = classifier.predict(X_test_scaled)
    
    def build_test_windows(data_scaled, lookback=10):
        windows = []
        for i in range(len(data_scaled)):
            if i < lookback:
                pad_needed = lookback - (i + 1)
                window = np.vstack([data_scaled[0:1]] * pad_needed + [data_scaled[0:i+1]])
            else:
                window = data_scaled[i-lookback+1:i+1]
            windows.append(window)
        return np.array(windows)
        
    X_test_3d = build_test_windows(X_test_scaled)
    y_pred_lstm = lstm_regressor.predict(X_test_3d).flatten()
    
    bt_df = pd.DataFrame(index=test_dates)
    bt_df['BTC_Return'] = actual_returns
    bt_df['CLF_Signal'] = y_pred_class
    bt_df['LSTM_Return_Pred'] = y_pred_lstm
    
    # XGBoost probabilities
    proba = classifier.predict_proba(X_test_scaled)
    bt_df['UP_Prob'] = proba[:, 1]

    # Position sizing
    bt_df['Position_Size'] = np.where( bt_df['UP_Prob'] > 0.50, 1.0, 0 )

    # Ensemble filter
    bt_df['Ensemble_Signal'] = np.where(
        (bt_df['Position_Size'] > 0) &
        (bt_df['LSTM_Return_Pred'] > 0),
        bt_df['Position_Size'],
        0
    )

    # Transaction costs
    TRANSACTION_COST = 0.001

    bt_df['Position_Change'] = (
        bt_df['Ensemble_Signal']
        .diff()
        .abs()
        .fillna(0)
        )

    bt_df['Cost'] = (
        bt_df['Position_Change']
        * TRANSACTION_COST
    )

    bt_df['Strat_Return'] = (
        bt_df['Ensemble_Signal']
        * bt_df['BTC_Return']
    ) - bt_df['Cost']
    
    bt_df['Cum_BTC'] = np.exp(bt_df['BTC_Return'].cumsum()) * 10000
    bt_df['Cum_Strat'] = np.exp(bt_df['Strat_Return'].cumsum()) * 10000
    return bt_df

bt_results = run_strategy_backtest(df_market)

st.subheader("Debug Statistics")

st.write("UP Probability")
st.write(bt_results["UP_Prob"].describe())

st.write("LSTM Predictions")
st.write(bt_results["LSTM_Return_Pred"].describe())

st.write("Signal Counts")
st.write(bt_results['Ensemble_Signal'].value_counts())

strategy_returns = bt_results['Strat_Return']

# Sharpe Ratio
if strategy_returns.std() != 0:
    sharpe_ratio = (
        strategy_returns.mean()
        / strategy_returns.std()
    ) * np.sqrt(252)
else:
    sharpe_ratio = 0

# CAGR
years = len(bt_results) / 252

if years > 0:
    cagr = (
        (bt_results['Cum_Strat'].iloc[-1] / 10000)
        ** (1 / years) - 1
    ) * 100
else:
    cagr = 0

# Drawdown
rolling_max = bt_results['Cum_Strat'].cummax()

drawdown = (
    bt_results['Cum_Strat']
    - rolling_max
) / rolling_max

max_drawdown = drawdown.min() * 100

# Win Rate
trade_days = bt_results[
    bt_results['Ensemble_Signal'] > 0
]

if len(trade_days) > 0:
    win_rate = (
        (trade_days['BTC_Return'] > 0).sum()
        / len(trade_days)
    ) * 100
else:
    win_rate = 0

num_trades = (
    bt_results['Position_Change'] > 0
).sum()

# 🛑 SAFETY CHECK 3: Verify backtest results generated data
if bt_results.empty:
    st.warning("⚠️ Backtest split resulted in 0 samples. Increase your historical dataset window or decrease your split percentage.")
    st.stop()

# 4. Live Forward Signals Module
features = [
    'Log_Returns', 'Log_Returns_Lag_2', 'Volume_Lag_1', 
    'Volatility_7d', 'Volume_Lag_2', 'Log_Returns_Lag_3', 
    'Log_Returns_Lag_1', 'Volume_Lag_3'
]

# Ensure we have at least 10 entries for the window
if len(df_market) >= 10:
    last_10_days = df_market[features].tail(10)
    scaled_input_2d = scaler.transform(last_10_days)
    current_day_flat = scaled_input_2d[-1].reshape(1, -1)
    current_window_3d = scaled_input_2d.reshape(1, 10, 8)

    live_direction = classifier.predict(current_day_flat)[0]
    live_confidence = classifier.predict_proba(current_day_flat)[0]
    live_lstm_return = lstm_regressor.predict(current_window_3d)[0][0]
else:
    st.error("🚨 Insufficient market history to construct live features.")
    st.stop()

# --- UI PRESENTATION LAYER ---
st.subheader("💡 Live Market Inferences (Tomorrow's Forecast)")
m1, m2, m3 = st.columns(3)

with m1:
    if live_direction == 1:
        st.metric("XGBoost Direction", "🟢 UP TREND", f"Confidence: {live_confidence[1]*100:.1f}%")
    else:
        st.metric("XGBoost Direction", "🔴 DOWN TREND", f"Confidence: {live_confidence[0]*100:.1f}%")

with m2:
    pct_move = (np.exp(live_lstm_return) - 1) * 100
    current_price = float(df_market['Close'].iloc[-1])
    target_usd = current_price * (1 + (pct_move / 100))
    st.metric("LSTM Magnitude", f"{pct_move:+.2f}%", f"Target: ${target_usd:,.2f}")

with m3:
    if live_direction == 1 and live_lstm_return > 0.005:
        st.metric("Ensemble Trade Action", "🔥 STRONG BUY", "Conditions Met")
    else:
        st.metric("Ensemble Trade Action", "🛡️ SIT IN CASH", "Defensive Stance")

st.markdown("---")
st.subheader("📊 Out-of-Sample Historical Strategy Performance")

st.subheader("📈 Professional Risk Metrics")

r1, r2, r3, r4 = st.columns(4)

r1.metric(
    "Sharpe Ratio",
    f"{sharpe_ratio:.2f}"
)

r2.metric(
    "Max Drawdown",
    f"{max_drawdown:.2f}%"
)

r3.metric(
    "CAGR",
    f"{cagr:.2f}%"
)

r4.metric(
    "Win Rate",
    f"{win_rate:.1f}%"
)

col_metric_1, col_metric_2 = st.columns(2)
final_btc_roi = (bt_results['Cum_BTC'].iloc[-1] / 10000 - 1) * 100
final_strat_roi = (bt_results['Cum_Strat'].iloc[-1] / 10000 - 1) * 100

col_metric_1.metric("Benchmark Bitcoin Return", f"{final_btc_roi:.2f}%")
col_metric_2.metric("Ensemble Strategy Return", f"{final_strat_roi:.2f}%", f"Alpha: {final_strat_roi - final_btc_roi:+.2f}%")

# Plotly Interactive Equity Curve
fig = go.Figure()
fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results['Cum_BTC'], name='Buy & Hold BTC', line=dict(color='#FF9900', width=2)))
fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results['Cum_Strat'], name='Ensemble AI Strategy', line=dict(color='#00FFCC', width=2.5)))

buy_signals = bt_results[
    bt_results['Position_Change'] > 0
]

sell_signals = bt_results[
    bt_results['Position_Change'] < 0
]

fig.update_layout(
    title='Growth of $10,000 Investment (Simulation Run Over Test Window)',
    xaxis_title='Timeline',
    yaxis_title='Portfolio Value ($)',
    template='plotly_dark',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)
st.plotly_chart(fig, use_container_width=True)
  