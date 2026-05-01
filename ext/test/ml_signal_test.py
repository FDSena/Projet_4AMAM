from ext.ml_signals import build_features,  build_features, build_target, split_train_test

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
"""
# --- Test ---
features = build_features(price_data, window=5)

print("Shape :", features.shape)
print("\nColonnes :", features.columns.tolist())
print("\nAperçu :\n", features.head())
print("\nValeurs manquantes :", features.isna().sum().sum())



# test fonction build_targets
returns_data = price_data.pct_change().dropna()

# Mode binary
target_binary = build_target(returns_data, horizon=1, mode="binary")
print("Binary shape :", target_binary.shape)
print("\nAperçu binary :\n", target_binary.head())


# Mode signal
target_signal = build_target(returns_data, horizon=1, mode="signal")
print("\nSignal shape :", target_signal.shape)
print("\nAperçu signal :\n", target_signal.head())

#Mode continuous
target_continuous = build_target(returns_data, horizon=1, mode="continuous")
print("\nContinuous shape :", target_continuous.shape)
print("\nAperçu continuous :\n", target_continuous.head())
# Mode invalide
try:
    build_target(returns_data, mode="invalid")
except ValueError as e:
    print("\nErreur attendue :", e)
"""

# Données
returns_data = price_data.pct_change().dropna()
features = build_features(price_data, window=5)
target = build_target(returns_data, horizon=1, mode="binary")

# Split
X_train, X_test, y_train, y_test = split_train_test(features, target, train_ratio=0.8)

print("X_train shape :", X_train.shape)
print("X_test shape  :", X_test.shape)
print("y_train shape :", y_train.shape)
print("y_test shape  :", y_test.shape)

print("\nDernière date train :", X_train.index[-1])
print("Première date test  :", X_test.index[0])
