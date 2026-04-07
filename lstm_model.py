"""LSTM model module — build, train, predict."""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def build_model(look_back: int = 60) -> Sequential:
    """Build a 2-layer LSTM model.

    Args:
        look_back: Number of timesteps (input shape).

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(model, X_train, y_train, epochs: int = 25, batch_size: int = 32):
    """Train the LSTM model with early stopping.

    Returns:
        Training history object.
    """
    early_stop = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )
    return history


def predict(model, X, scaler):
    """Generate predictions and inverse-transform to original scale.

    Returns:
        Predictions as a 1-D numpy array in original price scale.
    """
    preds = model.predict(X, verbose=0)
    return scaler.inverse_transform(preds).flatten()


def predict_future(model, last_sequence, scaler, days: int = 7):
    """Predict future stock prices iteratively.

    Args:
        model: Trained LSTM model.
        last_sequence: Last look_back-length scaled sequence (shape: (look_back,)).
        scaler: Fitted MinMaxScaler.
        days: Number of future days to predict.

    Returns:
        List of predicted prices for the next `days` days.
    """
    current = last_sequence.copy()
    predictions = []
    for _ in range(days):
        x_input = current.reshape(1, len(current), 1)
        pred = model.predict(x_input, verbose=0)[0, 0]
        predictions.append(pred)
        current = np.append(current[1:], pred)

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()
