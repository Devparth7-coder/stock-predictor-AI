# 📈 Stock Price Prediction Web App using LSTM

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> A deep learning–powered web application that forecasts stock prices using Long Short-Term Memory (LSTM) neural networks — with an interactive interface built on Streamlit and live data sourced from Yahoo Finance.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Features](#4-features)
5. [Tech Stack](#5-tech-stack)
6. [System Architecture](#6-system-architecture)
7. [Dataset Information](#7-dataset-information)
8. [Data Preprocessing](#8-data-preprocessing)
9. [Model Architecture](#9-model-architecture)
10. [Training Process](#10-training-process)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Results and Observations](#12-results-and-observations)
13. [Limitations](#13-limitations)
14. [Future Improvements](#14-future-improvements)
15. [Installation Guide](#15-installation-guide)
16. [Usage Instructions](#16-usage-instructions)
17. [Deployment](#17-deployment)
18. [Screenshots](#18-screenshots)
19. [Conclusion](#19-conclusion)

---

## 1. Project Overview

This project implements a **Stock Price Prediction Web Application** that leverages a Long Short-Term Memory (LSTM) deep learning model to forecast closing prices of publicly traded stocks. The application fetches historical stock data in real time using the `yfinance` API, preprocesses it, trains an LSTM model, and visualizes predicted versus actual price trends through an interactive Streamlit dashboard.

The project is designed to demonstrate the application of time-series forecasting using deep learning in the finance domain. It is intended for educational purposes and exploratory analysis — not for real-world investment decisions.

---

## 2. Problem Statement

Financial markets are inherently complex and non-linear. Traditional statistical models such as ARIMA or linear regression often fall short when capturing the long-range dependencies present in stock price time series. This project addresses that gap by applying LSTM-based deep learning, which is specifically designed to learn patterns in sequential data.

**Core challenge:** Given the past `N` days of a stock's closing price, can a model accurately predict the price for the next trading day?

---

## 3. Objectives

- Fetch and process real-time historical stock data from Yahoo Finance.
- Build and train an LSTM-based neural network for time-series forecasting.
- Evaluate the model's predictive performance using standard regression metrics.
- Visualize actual versus predicted stock prices on an interactive dashboard.
- Provide a clean, beginner-friendly web interface accessible to non-technical users.

---

## 4. Features

- **Real-Time Data Fetching** — Retrieves historical OHLCV data using `yfinance` for any valid stock ticker.
- **Data Normalization** — Applies `MinMaxScaler` to normalize price data for stable model training.
- **LSTM Deep Learning Model** — Multi-layer LSTM architecture trained on sliding-window sequences.
- **Train/Test Split** — Uses an 80/20 temporal split to prevent data leakage.
- **Prediction Visualization** — Matplotlib charts overlay actual vs. predicted prices.
- **Interactive Web Interface** — Built with Streamlit; supports dynamic ticker input and date range selection.
- **Model Persistence** — Trained models can be saved and reloaded to avoid retraining.

---

## 5. Tech Stack

| Tool / Library | Purpose |
|---|---|
| **Python 3.9+** | Core programming language |
| **TensorFlow / Keras** | Deep learning framework used to build and train the LSTM model |
| **Pandas** | Data manipulation and time-series structuring |
| **NumPy** | Numerical operations and array handling |
| **Scikit-learn** | Data preprocessing (MinMaxScaler) and evaluation metrics |
| **Matplotlib** | Static charting for actual vs. predicted price visualizations |
| **Streamlit** | Web application framework for building the interactive frontend |
| **yfinance** | Python wrapper for Yahoo Finance API to fetch historical stock data |

---

## 6. System Architecture

The application follows a linear data pipeline from raw market data to interactive predictions:

```
[User Input: Ticker Symbol]
          │
          ▼
[yfinance API] ──► Fetch historical OHLCV data
          │
          ▼
[Data Preprocessing]
  - Filter closing prices
  - Normalize with MinMaxScaler
  - Create sliding-window sequences (e.g., 60-day look-back)
          │
          ▼
[Train/Test Split (80/20)]
          │
          ▼
[LSTM Model Training]
  - 2–4 LSTM layers with Dropout regularization
  - Dense output layer (1 neuron)
  - Optimizer: Adam | Loss: Mean Squared Error
          │
          ▼
[Model Evaluation]
  - Inverse-transform predictions
  - Compute RMSE
          │
          ▼
[Streamlit Dashboard]
  - Display charts: Actual vs. Predicted
  - Show performance metrics
```

**Workflow Summary:**

1. The user enters a stock ticker (e.g., `AAPL`, `TSLA`) and selects a date range.
2. `yfinance` fetches daily closing price data for the given period.
3. The data is normalized and reshaped into 60-day input sequences.
4. The LSTM model trains on 80% of the data and predicts on the remaining 20%.
5. Predictions are inverse-transformed back to the original price scale.
6. Results are rendered as interactive Matplotlib charts within the Streamlit app.

---

## 7. Dataset Information

| Property | Details |
|---|---|
| **Source** | Yahoo Finance via `yfinance` Python library |
| **Data Type** | Historical daily stock prices (OHLCV) |
| **Primary Feature Used** | Closing Price (`Close`) |
| **Frequency** | Daily |
| **Typical Date Range** | Configurable (default: last 5–10 years) |
| **Format** | Pandas DataFrame |

**Example Tickers Supported:** `AAPL` (Apple), `GOOGL` (Google), `TSLA` (Tesla), `MSFT` (Microsoft), `AMZN` (Amazon), and any other valid Yahoo Finance ticker.

```python
import yfinance as yf

ticker = "AAPL"
df = yf.download(ticker, start="2015-01-01", end="2024-01-01")
data = df[['Close']]
```

---

## 8. Data Preprocessing

Proper preprocessing is critical for training stable LSTM models on financial time-series data.

### Steps:

**1. Feature Selection**
Only the `Close` price column is used as the input feature. OHLCV columns are dropped to keep the model focused.

**2. Normalization**
`MinMaxScaler` scales values to the range `[0, 1]`, which is essential for gradient-based optimization:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
```

**3. Sequence Creation (Sliding Window)**
A look-back window of 60 days is used. Each input sample `X` contains 60 consecutive days of normalized prices; the target `y` is the price on day 61:

```python
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))  # [samples, timesteps, features]
```

**4. Train/Test Split**
An 80/20 temporal split is applied. Random shuffling is deliberately avoided to preserve time ordering:

```python
split = int(len(X) * 0.80)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
```

---

## 9. Model Architecture

### What is LSTM?

Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) designed to learn long-range dependencies in sequential data. Unlike standard RNNs, LSTMs use a **gating mechanism** (input gate, forget gate, output gate) to control information flow, allowing them to retain relevant context across hundreds of time steps without suffering from the vanishing gradient problem.

This makes LSTMs well-suited for financial time-series forecasting, where price patterns may span weeks or months.

### Model Definition

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),

    LSTM(units=50, return_sequences=True),
    Dropout(0.2),

    LSTM(units=50, return_sequences=False),
    Dropout(0.2),

    Dense(units=25),
    Dense(units=1)  # Output: next day's closing price
])
```

### Architecture Summary

| Layer | Units | Notes |
|---|---|---|
| LSTM (1) | 50 | Returns sequences; accepts 60-timestep input |
| Dropout | 20% | Reduces overfitting |
| LSTM (2) | 50 | Returns sequences |
| Dropout | 20% | Reduces overfitting |
| LSTM (3) | 50 | Does not return sequences |
| Dropout | 20% | Reduces overfitting |
| Dense | 25 | Intermediate fully connected layer |
| Dense (Output) | 1 | Predicts next closing price |

---

## 10. Training Process

```python
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)
```

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam | Adaptive learning rate; works well for RNNs |
| Loss Function | Mean Squared Error (MSE) | Standard for regression tasks |
| Epochs | 25 | Sufficient for convergence on most datasets |
| Batch Size | 32 | Balances training speed and gradient stability |
| Validation Split | 10% | Monitors generalization during training |

**Model Saving:**

```python
model.save("lstm_stock_model.h5")
```

---

## 11. Evaluation Metrics

### Root Mean Squared Error (RMSE)

RMSE is the primary evaluation metric. It penalizes large prediction errors and is expressed in the same unit as the original price (USD), making it highly interpretable:

```python
from sklearn.metrics import mean_squared_error
import numpy as np

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
print(f"RMSE: {rmse:.2f}")
```

### Training Loss

The training and validation loss curves (MSE over epochs) are plotted to detect overfitting or underfitting:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()
```

---

## 12. Results and Observations

- The LSTM model generally captures **macro-level price trends** well — including sustained uptrends, downtrends, and gradual recoveries.
- For stable, high-volume stocks like `AAPL` and `MSFT`, the model produces visually close predictions during normal market conditions.
- Short-term volatility and sudden price spikes (e.g., earnings announcements, macro events) are harder to predict accurately.
- RMSE values vary by stock and time period; lower RMSE indicates tighter prediction accuracy.
- Training loss typically converges within 15–25 epochs on standard datasets.

> **Note:** These results are for informational and educational purposes only. Model accuracy on historical data does not imply future predictive reliability.

---

## 13. Limitations

This section is critical for an honest and credible project presentation.

- **Stock markets are inherently unpredictable.** No model — regardless of complexity — can reliably forecast future stock prices. External factors such as geopolitical events, regulatory changes, and macroeconomic shifts are not captured in historical price data.
- **LSTM predicts patterns, not fundamentals.** The model learns statistical patterns from past prices and cannot account for earnings reports, news sentiment, or company-specific events.
- **Overfitting risk on short datasets.** Tickers with limited trading history may result in models that memorize training data rather than generalizing.
- **Lag effect.** LSTM predictions often track the actual price with a slight delay, meaning the model is effectively learning to "echo" recent prices rather than truly forecasting ahead.
- **Single feature input.** Using only the closing price discards potentially valuable signals such as volume, moving averages, or technical indicators.
- **No live trading integration.** This application does not connect to any brokerage or execute trades. It is purely a research and visualization tool.

---

## 14. Future Improvements

- **Multi-feature input:** Incorporate trading volume, RSI, MACD, Bollinger Bands, and other technical indicators as additional input features.
- **Sentiment analysis integration:** Combine news headlines and social media sentiment (via NLP) with price data for more context-aware predictions.
- **Hyperparameter optimization:** Use tools like Keras Tuner or Optuna to systematically tune LSTM units, dropout rates, look-back windows, and learning rates.
- **Alternative architectures:** Experiment with GRU (Gated Recurrent Unit), Temporal Convolutional Networks (TCN), or Transformer-based models for time series.
- **Confidence intervals:** Implement Monte Carlo Dropout or Bayesian approaches to generate prediction uncertainty bands.
- **Multi-step forecasting:** Extend predictions beyond the next single day (e.g., 5-day or 30-day horizon).
- **Portfolio-level analysis:** Expand the tool to analyze and compare multiple tickers simultaneously.
- **User authentication and saved sessions:** Allow users to save model runs and compare predictions over time.

---

## 15. Installation Guide

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```
tensorflow>=2.10.0
keras>=2.10.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
matplotlib>=3.6.0
streamlit>=1.20.0
yfinance>=0.2.18
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow, streamlit, yfinance; print('All packages installed successfully.')"
```

---

## 16. Usage Instructions

### Run the Streamlit Application

```bash
streamlit run app.py
```

This launches the web application in your default browser at `http://localhost:8501`.

### Using the Application

1. **Enter Ticker Symbol** — Type a valid stock ticker (e.g., `AAPL`, `TSLA`, `INFY.NS` for Indian stocks).
2. **Select Date Range** — Use the date pickers to define the historical data window.
3. **Click "Predict"** — The app fetches data, preprocesses it, trains the LSTM model, and generates predictions.
4. **View Results** — A chart displays actual vs. predicted prices. RMSE is shown below the chart.

### Running the Model Standalone (Without UI)

```bash
python model/train.py --ticker AAPL --start 2015-01-01 --end 2024-01-01
```

---

## 17. Deployment

### Option 1: Streamlit Community Cloud (Free)

1. Push your project to a public GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, select your repository, branch, and `app.py` as the entry point.
4. Click **"Deploy"** — your app will be live within minutes.

> Ensure `requirements.txt` is present in the root directory.

### Option 2: Render

1. Create a free account at [render.com](https://render.com).
2. Create a new **Web Service** and connect your GitHub repository.
3. Set the **Start Command** to:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
4. Choose the free tier and click **Deploy**.

### Option 3: Docker (Self-Hosted)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t lstm-stock-app .
docker run -p 8501:8501 lstm-stock-app
```

---

## 18. Screenshots

### Application Home Screen
> *Insert screenshot here — Streamlit home with ticker input and date selectors*

---

### Actual vs. Predicted Price Chart
> *Insert screenshot here — Matplotlib line chart overlaying actual (blue) and predicted (orange) closing prices*

---

### Training Loss Curve
> *Insert screenshot here — Epoch vs. MSE loss plot showing training and validation loss convergence*

---

### RMSE and Performance Metrics Panel
> *Insert screenshot here — Streamlit metric cards displaying RMSE, start date, end date, and data shape*

---

## 19. Conclusion

This project demonstrates a practical application of LSTM-based deep learning for financial time-series forecasting. By combining real-time data retrieval, robust preprocessing, a multi-layer LSTM architecture, and an interactive web interface, the application provides a complete, end-to-end pipeline for stock price analysis.

Key takeaways:

- LSTM networks are effective at learning temporal dependencies in stock price sequences.
- Proper normalization and sliding-window sequencing are essential for stable training.
- Prediction accuracy is reasonable for trend-following but limited during high-volatility periods.
- The inherent unpredictability of financial markets means this tool should be treated as an educational and exploratory resource — not a financial advisor.

This project serves as a strong foundation for further research in quantitative finance, time-series deep learning, and financial data engineering.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com/) for providing accessible financial data.
- [TensorFlow/Keras](https://www.tensorflow.org/) for the deep learning framework.
- [Streamlit](https://streamlit.io/) for making web app development effortless in Python.
- The open-source community for libraries that made this project possible.

---

*Built with ❤️ using Python, TensorFlow, and Streamlit.*
