"""Stock Price Prediction Web App using LSTM — main Streamlit application."""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

from utils.data_fetcher import fetch_stock_data, validate_ticker
from utils.preprocessor import preprocess
from model.lstm_model import build_model, train_model, predict, predict_future
from utils.visualizer import plot_actual_vs_predicted, plot_future

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

# ── Dark-mode custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: #1a1d23;
        border: 1px solid #2d333b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card h3 { color: #8b949e; font-size: 14px; margin: 0; }
    .metric-card p { color: #f0f6fc; font-size: 28px; font-weight: 700; margin: 4px 0 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Stock Price Prediction")
st.caption("LSTM-powered stock price forecasting with real-time data")

# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    ticker = st.text_input("Stock Ticker", value="AAPL", max_chars=10).upper().strip()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=3 * 365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    look_back = st.slider("Look-back Window (days)", 30, 120, 60)
    epochs = st.slider("Training Epochs", 5, 50, 25)
    future_days = st.slider("Future Prediction (days)", 1, 30, 7)

    predict_btn = st.button("🚀 Predict", use_container_width=True, type="primary")

# ── Main Logic ────────────────────────────────────────────────────────────────
if predict_btn:
    # Validate ticker
    if not ticker:
        st.error("Please enter a stock ticker symbol.")
        st.stop()

    with st.spinner("Validating ticker..."):
        if not validate_ticker(ticker):
            st.error(f"❌ Invalid ticker **{ticker}**. Please check and try again.")
            st.stop()

    # Fetch data
    with st.spinner(f"Fetching data for {ticker}..."):
        df = fetch_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if df.empty:
        st.error("No data returned. Try a different date range or ticker.")
        st.stop()

    if len(df) < look_back + 20:
        st.error(f"Not enough data points ({len(df)}). Need at least {look_back + 20}. Try a wider date range.")
        st.stop()

    # Show latest data
    st.subheader(f"📊 {ticker} — Latest Data")
    st.dataframe(df.tail(10).style.format("{:.2f}"), use_container_width=True)

    # Metrics cards
    latest = df["Close"].iloc[-1]
    change = df["Close"].iloc[-1] - df["Close"].iloc[-2]
    pct = (change / df["Close"].iloc[-2]) * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${latest:.2f}")
    c2.metric("Change", f"${change:.2f}", f"{pct:+.2f}%")
    c3.metric("52w High", f"${df['Close'].tail(252).max():.2f}")
    c4.metric("52w Low", f"${df['Close'].tail(252).min():.2f}")

    # Preprocess
    with st.spinner("Preprocessing data..."):
        X_train, y_train, X_test, y_test, scaler, train_size = preprocess(df, look_back)

    # Train
    with st.spinner("Training LSTM model... This may take a minute."):
        model = build_model(look_back)
        history = train_model(model, X_train, y_train, epochs=epochs)

    # Predict on test set
    with st.spinner("Generating predictions..."):
        train_preds = predict(model, X_train, scaler)
        test_preds = predict(model, X_test, scaler)

        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, test_preds))
    final_loss = history.history["loss"][-1]

    st.subheader("📉 Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test RMSE", f"${rmse:.2f}")
    m2.metric("Final Training Loss", f"{final_loss:.6f}")
    m3.metric("Epochs Trained", len(history.history["loss"]))

    # Actual vs Predicted chart
    test_dates = df.index[look_back + train_size :]
    fig1 = plot_actual_vs_predicted(test_dates, y_test_actual, test_preds, ticker)
    st.plotly_chart(fig1, use_container_width=True)

    # Future predictions
    st.subheader(f"🔮 Future Prediction — Next {future_days} Days")
    with st.spinner("Predicting future prices..."):
        close_scaled = scaler.transform(df["Close"].values.reshape(-1, 1)).flatten()
        last_seq = close_scaled[-look_back:]
        future_preds = predict_future(model, last_seq, scaler, days=future_days)

        last_date = df.index[-1]
        future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=future_days)

    fig2 = plot_future(df.index[-90:], df["Close"].values[-90:], future_dates, future_preds, ticker)
    st.plotly_chart(fig2, use_container_width=True)

    # Future prices table
    future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_preds})
    future_df["Date"] = future_df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(future_df.style.format({"Predicted Price": "${:.2f}"}), use_container_width=True)

    st.success("✅ Prediction complete!")

else:
    st.info("👈 Configure settings in the sidebar and click **Predict** to get started.")
