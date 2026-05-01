from ml_signals import build_features

import numpy as np
import pandas as pd

# --- Données factices ---
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
assets = ["AAPL", "MSFT", "GOOGL"]

price_data = pd.DataFrame(
    data=100 * np.cumprod(1 + np.random.randn(100, 3) * 0.01, axis=0),
    index=dates,
    columns=assets
)

# --- Test ---
features = build_features(price_data, window=5)

print("Shape :", features.shape)
print("\nColonnes :", features.columns.tolist())
print("\nAperçu :\n", features.head())
print("\nValeurs manquantes :", features.isna().sum().sum())



