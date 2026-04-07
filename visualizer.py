"""Visualization module — Plotly charts for stock predictions."""

import plotly.graph_objects as go
import pandas as pd


def plot_actual_vs_predicted(dates, actual, predicted, ticker: str):
    """Create an interactive Plotly chart comparing actual vs predicted prices."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode="lines", name="Actual", line=dict(color="#00d4aa", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=predicted, mode="lines", name="Predicted", line=dict(color="#ff6b6b", width=2)))
    fig.update_layout(
        title=f"{ticker} — Actual vs Predicted Closing Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def plot_future(dates, prices, future_dates, future_prices, ticker: str):
    """Plot historical prices with future prediction overlay."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines", name="Historical", line=dict(color="#00d4aa", width=2)))
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_prices, mode="lines+markers",
        name="Future Prediction", line=dict(color="#ffd93d", width=2, dash="dash"),
        marker=dict(size=6),
    ))
    fig.update_layout(
        title=f"{ticker} — Future Price Prediction (Next {len(future_prices)} Days)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
