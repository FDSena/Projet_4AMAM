from ext.ml_signals import build_features, build_target, simple_signal_model, split_train_test, predict_signal, compute_signal_loss, build_ml_signals
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


# Données
features = build_features(price_data, window=5)

# Signal
signals = simple_signal_model(features, threshold=0.01)


print("Shape :", signals.shape)
print("\nAperçu :\n", signals.head(5))
print("\nDistribution des signaux :\n", signals.value_counts())
"""

# Construction des features
features = build_features(price_data, window=5)

# Test predict_signal avec simple_signal_model
signals = predict_signal(
    model=simple_signal_model,
    features=features,
    threshold=0.01
)

print("Shape signals :", signals.shape)
print("\nAperçu signals :")
print(signals.head())

print("\nDistribution des signaux :")
print(signals.value_counts())

print("\nIndex aligné avec features :", signals.index.equals(features.index))
print("Longueur correcte :", len(signals) == len(features))

from ext.ml_signals import build_features, simple_signal_model, signal_to_weights

import numpy as np
import pandas as pd

# 1. Features
features = build_features(price_data, window=5)

# 2. Signal
signals = simple_signal_model(features, threshold=0.01)

# 3. Weights
weights = signal_to_weights(signals, n_assets=3)

# --- PRINT ---
print("Shape weights :", weights.shape)

print("\nAperçu weights :")
print(weights.head())

print("\nSomme des poids (doit être 1 ou 0) :")
print(weights.sum(axis=1).head())

print("\nExemple correspondance signal → poids :")
for i in range(5):
    date = signals.index[i]
    print(date, "Signal:", signals.iloc[i], "→ Weights:", weights.iloc[i].values)
    
# Rendements réalisés
returns_data = price_data.pct_change().dropna()

# On prend les rendements AAPL, alignés avec les signaux
realized_returns = returns_data["AAPL"]

# Test de la loss
loss_simple = compute_signal_loss(signals, realized_returns, mode="simple")
loss_penalty = compute_signal_loss(signals, realized_returns, mode="penalty")
loss_risk = compute_signal_loss(signals, realized_returns, mode="risk")

print("\nLoss simple :", loss_simple)
print("Loss penalty :", loss_penalty)
print("Loss risk :", loss_risk)

results = build_ml_signals(
    price_data=price_data,
    window=5,
    threshold=0.01
)

print(results.keys())

print("\nFeatures :")
print(results["features"].head())

print("\nSignals :")
print(results["signals"].head())

print("\nWeights :")
print(results["weights"].head())

print(results["signals"].value_counts())
print(results["weights"][results["weights"].sum(axis=1) > 0].head())