"""
ml_signals.py

Module de génération de signaux d'investissement et de poids dynamiques
pour une stratégie de portefeuille multi-actifs.

Logique financière :
- construire des features à partir des prix et rendements
- calculer un score par actif
- convertir ces scores en poids de portefeuille
- produire des allocations dynamiques compatibles avec le backtest
"""

import numpy as np
import pandas as pd


# ============================================================
# 1. VALIDATION DES DONNEES
# ============================================================

def ensure_dataframe(data, name="data"):
    """
    Convertir une Series ou array-like en DataFrame.
    """
    if isinstance(data, pd.Series):
        return data.to_frame()

    if isinstance(data, pd.DataFrame):
        return data.copy()

    try:
        return pd.DataFrame(data)
    except Exception as e:
        raise ValueError(f"{name} cannot be converted to a DataFrame.") from e


# ============================================================
# 2. CALCUL DES RENDEMENTS
# ============================================================

def compute_returns(price_data):
    """
    Calculer les rendements simples à partir des prix.
    """
    price_data = ensure_dataframe(price_data, "price_data")

    returns = price_data.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)

    return returns


# ============================================================
# 3. CONSTRUCTION DES FEATURES
# ============================================================

def build_features(price_data, returns_data=None, window=20):
    """
    Construire des variables explicatives par actif.

    Features :
    - momentum
    - rendement moyen glissant
    - volatilité glissante
    - position du prix par rapport à sa moyenne mobile
    - score rendement / volatilité
    """
    price_data = ensure_dataframe(price_data, "price_data")

    if returns_data is None:
        returns_data = compute_returns(price_data)
    else:
        returns_data = ensure_dataframe(returns_data, "returns_data")

    if window <= 1:
        raise ValueError("window must be greater than 1.")

    # Momentum : performance sur une fenêtre donnée
    momentum = price_data / price_data.shift(window) - 1

    # Rendement moyen glissant
    rolling_mean_return = returns_data.rolling(window).mean()

    # Volatilité glissante
    rolling_volatility = returns_data.rolling(window).std()

    # Moyenne mobile
    moving_average = price_data.rolling(window).mean()
    price_vs_ma = (price_data - moving_average) / moving_average

    # Score rendement / risque
    risk_adjusted_score = rolling_mean_return / rolling_volatility

    features = {
        "momentum": momentum,
        "rolling_mean_return": rolling_mean_return,
        "rolling_volatility": rolling_volatility,
        "price_vs_ma": price_vs_ma,
        "risk_adjusted_score": risk_adjusted_score,
    }

    # Nettoyage
    for key in features:
        features[key] = features[key].replace([np.inf, -np.inf], np.nan)

    return features


# ============================================================
# 4. CONSTRUCTION D'UN SCORE FINANCIER PAR ACTIF
# ============================================================

def build_asset_scores(
    price_data,
    returns_data=None,
    window=20,
    momentum_weight=0.5,
    return_weight=0.3,
    risk_weight=0.2
):
    """
    Construire un score par actif.

    Idée :
    - un actif avec momentum positif est favorisé
    - un actif avec rendement moyen positif est favorisé
    - un actif très volatil est pénalisé

    Score simplifié :
    score = momentum_weight * momentum
          + return_weight * rolling_mean_return
          - risk_weight * rolling_volatility
    """
    features = build_features(price_data, returns_data, window)

    momentum = features["momentum"]
    rolling_mean_return = features["rolling_mean_return"]
    rolling_volatility = features["rolling_volatility"]

    scores = (
        momentum_weight * momentum
        + return_weight * rolling_mean_return
        - risk_weight * rolling_volatility
    )

    scores = scores.replace([np.inf, -np.inf], np.nan)

    return scores


# ============================================================
# 5. VERSION SCORE MOMENTUM / VOLATILITE
# ============================================================

def build_momentum_volatility_scores(price_data, returns_data=None, window=20):
    """
    Construire un score de type momentum / volatilité.

    Logique :
    - momentum élevé = actif attractif
    - volatilité élevée = actif risqué
    - donc on favorise les actifs avec bon momentum relativement au risque
    """
    price_data = ensure_dataframe(price_data, "price_data")

    if returns_data is None:
        returns_data = compute_returns(price_data)
    else:
        returns_data = ensure_dataframe(returns_data, "returns_data")

    momentum = price_data / price_data.shift(window) - 1
    volatility = returns_data.rolling(window).std()

    scores = momentum / volatility
    scores = scores.replace([np.inf, -np.inf], np.nan)

    return scores


# ============================================================
# 6. CONVERSION DES SCORES EN POIDS
# ============================================================

def scores_to_weights(scores, long_only=True, fallback="equal"):
    """
    Convertir des scores par actif en poids de portefeuille.

    Paramètres
    ----------
    scores : DataFrame
        Scores par actif et par date.

    long_only : bool
        Si True, les scores négatifs sont remplacés par 0.
        Cela interdit la vente à découvert.

    fallback : str
        Stratégie si tous les scores sont nuls sur une date.
        - "equal" : portefeuille équipondéré
        - "zero" : aucun investissement

    Retour
    ------
    weights : DataFrame
        Poids dynamiques du portefeuille.
    """
    scores = ensure_dataframe(scores, "scores")

    if long_only:
        positive_scores = scores.clip(lower=0)
    else:
        positive_scores = scores.copy()

    if long_only :
        row_sums =positive_scores.sum(axis=1)
    else:
        row_sums = positive_scores.abs().sum(axis=1)

    weights = positive_scores.div(row_sums.replace(0, np.nan), axis=0)

    if fallback == "equal":
        equal_weights = 1.0 / scores.shape[1]
        weights = weights.fillna(equal_weights)

    elif fallback == "zero":
        weights = weights.fillna(0.0)

    else:
        raise ValueError("fallback must be 'equal' or 'zero'.")

    return weights


# ============================================================
# 7. SIGNAL DISCRET PAR ACTIF
# ============================================================

def scores_to_signals(scores, threshold=0.0):
    """
    Transformer les scores en signaux discrets.

    Sortie :
    - 1 : actif attractif
    - 0 : neutre
    - -1 : actif défavorable
    """
    scores = ensure_dataframe(scores, "scores")

    signals = pd.DataFrame(0, index=scores.index, columns=scores.columns)

    signals[scores > threshold] = 1
    signals[scores < -threshold] = -1

    return signals


# ============================================================
# 8. CONSTRUCTION DE CIBLE POUR ML OPTIONNEL
# ============================================================

def build_target(returns_data, horizon=1, mode="binary"):
    """
    Construire une cible future pour un modèle supervisé éventuel.

    Modes :
    - binary : 1 si rendement futur positif, 0 sinon
    - continuous : rendement futur
    - signal : -1, 0 ou 1 selon le signe du rendement futur
    """
    returns_data = ensure_dataframe(returns_data, "returns_data")

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    future_return = returns_data.shift(-horizon)

    if mode == "binary":
        target = (future_return > 0).astype(int)

    elif mode == "continuous":
        target = future_return

    elif mode == "signal":
        target = np.sign(future_return)

    else:
        raise ValueError("mode must be 'binary', 'continuous', or 'signal'.")

    return target.dropna()


# ============================================================
# 9. ALIGNEMENT AVEC LES RENDEMENTS
# ============================================================

def align_weights_with_returns(weights, returns_data):
    """
    Aligner les poids dynamiques avec les rendements utilisés en backtest.

    Important :
    Les poids calculés à la date t doivent être appliqués au rendement futur.
    On décale donc les poids d'une période pour limiter la fuite temporelle.
    """

    weights = ensure_dataframe(weights, "weights")
    returns_data = ensure_dataframe(returns_data, "returns_data")

    if list(weights.columns) != list(returns_data.columns):
        raise ValueError("weights and returns_data must have the same columns.")


    shifted_weights = weights.shift(1)


    common_index = shifted_weights.index.intersection(returns_data.index)
    shifted_weights = shifted_weights.loc[common_index]
    returns_aligned = returns_data.loc[common_index]

    shifted_weights = shifted_weights.dropna(how='all')
    returns_aligned = returns_aligned.loc[shifted_weights.index]

    if len(shifted_weights) == 0:
        raise ValueError("No valid weights after shifting. Check the input data.")

    return shifted_weights, returns_aligned


# ============================================================
# 10. PIPELINE COMPLET
# ============================================================

def build_ml_signals(
    price_data,
    returns_data=None,
    window=20,
    method="momentum_volatility",
    long_only=True,
    fallback="equal",
    threshold=0.0,
    align_for_backtest=True
):
    """
    Pipeline complet de génération de signaux et poids dynamiques.

    Paramètres
    ----------
    price_data : DataFrame
        Prix des actifs.

    returns_data : DataFrame or None
        Rendements des actifs. Si None, ils sont calculés.

    window : int
        Fenêtre utilisée pour les indicateurs glissants.

    method : str
        Méthode de construction du score :
        - "momentum_volatility"
        - "financial_score"

    long_only : bool
        Interdit ou non les poids négatifs.

    fallback : str
        Que faire si aucun actif n'a un score positif :
        - "equal"
        - "zero"

    threshold : float
        Seuil utilisé pour les signaux discrets.

    align_for_backtest : bool
        Si True, décale les poids d'une période pour éviter la fuite temporelle.

    Retour
    ------
    results : dict
        Contient :
        - features
        - scores
        - signals
        - weights
        - weights_for_backtest
        - returns_for_backtest
    """
    price_data = ensure_dataframe(price_data, "price_data")

    if returns_data is None:
        returns_data = compute_returns(price_data)
    else:
        returns_data = ensure_dataframe(returns_data, "returns_data")

    features = build_features(
        price_data=price_data,
        returns_data=returns_data,
        window=window
    )

    if method == "momentum_volatility":
        scores = build_momentum_volatility_scores(
            price_data=price_data,
            returns_data=returns_data,
            window=window
        )

    elif method == "financial_score":
        scores = build_asset_scores(
            price_data=price_data,
            returns_data=returns_data,
            window=window
        )

    else:
        raise ValueError("method must be 'momentum_volatility' or 'financial_score'.")

    signals = scores_to_signals(scores, threshold=threshold)

    weights = scores_to_weights(
        scores=scores,
        long_only=long_only,
        fallback=fallback
    )

    results = {
        "features": features,
        "scores": scores,
        "signals": signals,
        "weights": weights
    }

    if align_for_backtest:
        weights_for_backtest, returns_for_backtest = align_weights_with_returns(
            weights=weights,
            returns_data=returns_data
        )

        results["weights_for_backtest"] = weights_for_backtest
        results["returns_for_backtest"] = returns_for_backtest

    return results