"""Preprocessing module — handles missing values and scaling."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess(df: pd.DataFrame, look_back: int = 60):
    """Preprocess stock data for LSTM training.

    Args:
        df: Raw stock DataFrame (must have 'Close' column).
        look_back: Number of past days used to predict next day.

    Returns:
        X_train, y_train, X_test, y_test, scaler, train_size
    """
    # Handle missing values
    df = df.copy()
    df["Close"] = df["Close"].ffill().bfill()

    close_prices = df["Close"].values.reshape(-1, 1)

    # Scale to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    # Create sequences
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back : i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)

    # Train/test split (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape for LSTM [samples, timesteps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, y_train, X_test, y_test, scaler, train_size
