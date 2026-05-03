"""
ml_signals.py
=============
Génération de signaux d'investissement et de poids dynamiques.

Fondements théoriques
---------------------
On construit des scores par actif à partir d'indicateurs
statistiques glissants, puis on les convertit en poids.

Indicateurs utilisés
--------------------
Momentum (fenêtre w) :
    mom_t = P_t / P_{t-w} − 1      (rendement sur la fenêtre)

Rendement moyen glissant :
    µ̂_t = (1/w) Σ_{s=t-w+1}^{t} r_s

Volatilité glissante :
    σ̂_t = std_{w}(r_s)             (Bessel, ddof=1)

Prix vs moyenne mobile :
    δ_t = (P_t − MA_t) / MA_t

Score rendement/risque (Sharpe simplifié) :
    SR_t = µ̂_t / σ̂_t

Score composite pondéré :
    score_t = α·mom_t + β·µ̂_t − γ·σ̂_t

Conversion en poids (long-only)
--------------------------------
Pour chaque date t :
    - on conserve les scores positifs  (s_i = max(score_i, 0))
    - on normalise :  w_i = s_i / Σ_j s_j

Si tous les scores sont négatifs → fallback "equal" ou "zero".

Biais look-ahead
----------------
Les poids calculés à la date t utilisent de l'information jusqu'à t.
Pour les appliquer au rendement r_{t+1} (non observé à t),
on décale les poids d'une période AVANT de les passer au backtest.
"""

import numpy as np
import pandas as pd


# ============================================================
# 1. UTILITAIRES
# ============================================================

def ensure_dataframe(data, name: str = "data") -> pd.DataFrame:
    """
    Convertir une entrée en DataFrame (copie défensive).
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame()
    try:
        return pd.DataFrame(data)
    except Exception as e:
        raise ValueError(f"'{name}' ne peut pas être converti en DataFrame.") from e


# ============================================================
# 2. CALCUL DES RENDEMENTS SIMPLES
# ============================================================

def compute_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculer les rendements simples r_t = (P_t − P_{t-1}) / P_{t-1}.

    La première ligne est NaN (pas de rendement à t=0).
    Les Inf sont remplacés par NaN.
    """
    price_data = ensure_dataframe(price_data, "price_data")
    returns    = price_data.pct_change()
    return returns.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 3. CONSTRUCTION DES FEATURES STATISTIQUES
# ============================================================

def build_features(
    price_data: pd.DataFrame,
    returns_data: pd.DataFrame | None = None,
    window: int = 20
) -> dict[str, pd.DataFrame]:
    """
    Calculer les indicateurs statistiques glissants sur la fenêtre w.

    Indicateurs retournés
    ---------------------
    - momentum           : P_t / P_{t-w} − 1
    - rolling_mean_return: µ̂_t  = mean_{w}(r)
    - rolling_volatility : σ̂_t  = std_{w}(r),  ddof=1
    - price_vs_ma        : (P_t − MA_t) / MA_t
    - risk_adjusted_score: µ̂_t / σ̂_t  (Sharpe glissant)

    Paramètres
    ----------
    price_data : (T, n) DataFrame — prix
    returns_data : (T, n) ou None — si None, calculé automatiquement
    window : int > 1

    Retour
    ------
    dict[str, DataFrame]
    """
    price_data = ensure_dataframe(price_data, "price_data")
    if returns_data is None:
        returns_data = compute_returns(price_data)
    else:
        returns_data = ensure_dataframe(returns_data, "returns_data")

    if window <= 1:
        raise ValueError("window doit être > 1.")

    momentum             = price_data / price_data.shift(window) - 1.0
    rolling_mean_return  = returns_data.rolling(window, min_periods=window).mean()
    rolling_volatility   = returns_data.rolling(window, min_periods=window).std(ddof=1)
    moving_average       = price_data.rolling(window, min_periods=window).mean()
    price_vs_ma          = (price_data - moving_average) / moving_average
    risk_adjusted_score  = rolling_mean_return / rolling_volatility.replace(0.0, np.nan)

    features = {
        "momentum":            momentum,
        "rolling_mean_return": rolling_mean_return,
        "rolling_volatility":  rolling_volatility,
        "price_vs_ma":         price_vs_ma,
        "risk_adjusted_score": risk_adjusted_score,
    }
    # Nettoyage numérique
    for k in features:
        features[k] = features[k].replace([np.inf, -np.inf], np.nan)

    return features


# ============================================================
# 4. SCORE COMPOSITE PONDÉRÉ
# ============================================================

def build_asset_scores(
    price_data: pd.DataFrame,
    returns_data: pd.DataFrame | None = None,
    window: int = 20,
    momentum_weight: float = 0.5,
    return_weight: float = 0.3,
    risk_weight: float = 0.2
) -> pd.DataFrame:
    """
    Calculer un score composite par actif et par date :

        score = α · mom  +  β · µ̂  −  γ · σ̂

    avec α + β + γ = 1 (convention de normalisation des poids).

    Interprétation financière :
    - momentum positif  → actif en tendance haussière (+)
    - rendement moyen positif → historique favorable    (+)
    - volatilité élevée → actif risqué, pénalisé        (−)

    Paramètres
    ----------
    momentum_weight : α ≥ 0
    return_weight   : β ≥ 0
    risk_weight     : γ ≥ 0  (pénalité sur la volatilité)

    Retour
    ------
    scores : (T, n) DataFrame
    """
    feat = build_features(price_data, returns_data, window)
    scores = (
        momentum_weight * feat["momentum"]
        + return_weight  * feat["rolling_mean_return"]
        - risk_weight    * feat["rolling_volatility"]
    )
    return scores.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 5. SCORE MOMENTUM / VOLATILITÉ (SHARPE GLISSANT)
# ============================================================

def build_momentum_volatility_scores(
    price_data: pd.DataFrame,
    returns_data: pd.DataFrame | None = None,
    window: int = 20
) -> pd.DataFrame:
    """
    Score de type Sharpe glissant :

        score_t = mom_t / σ̂_t

    Favorise les actifs avec fort momentum relativement à leur risque.
    Correspond à une maximisation du ratio de Sharpe instantané.
    Si σ̂_t ≈ 0, le score est mis à NaN (indéfini).

    Paramètres
    ----------
    price_data : (T, n)
    returns_data : (T, n) or None
    window : int

    Retour
    ------
    scores : (T, n) DataFrame
    """
    price_data = ensure_dataframe(price_data, "price_data")
    if returns_data is None:
        returns_data = compute_returns(price_data)
    else:
        returns_data = ensure_dataframe(returns_data, "returns_data")

    momentum   = price_data / price_data.shift(window) - 1.0
    volatility = returns_data.rolling(window, min_periods=window).std(ddof=1)

    scores = momentum / volatility.replace(0.0, np.nan)
    return scores.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 6. CONVERSION DES SCORES EN POIDS
# ============================================================

def scores_to_weights(
    scores: pd.DataFrame,
    long_only: bool = True,
    fallback: str = "equal"
) -> pd.DataFrame:
    """
    Convertir les scores en poids de portefeuille normalisés.

    Méthode long-only (long_only=True)
    -----------------------------------
    1. Clip : s_i ← max(s_i, 0)
    2. Normalise : w_i = s_i / Σ_j s_j

    Méthode long-short (long_only=False)
    -------------------------------------
    1. Normalise par la somme des valeurs absolues :
       w_i = s_i / Σ_j |s_j|
    → les actifs avec score négatif sont vendus à découvert.

    Fallback (ligne avec somme = 0 ou NaN complet)
    -----------------------------------------------
    "equal" : poids égaux 1/n  (portefeuille non informatif)
    "zero"  : poids nuls       (pas d'investissement ce jour-là)

    Paramètres
    ----------
    scores : (T, n) DataFrame
    long_only : bool
    fallback : "equal" ou "zero"

    Retour
    ------
    weights : (T, n) DataFrame, chaque ligne somme à 1 (ou 0 si fallback="zero")
    """
    scores = ensure_dataframe(scores, "scores")
    n = scores.shape[1]

    if long_only:
        s    = scores.clip(lower=0.0)
        denom = s.sum(axis=1).replace(0.0, np.nan)
    else:
        s    = scores.copy()
        denom = s.abs().sum(axis=1).replace(0.0, np.nan)

    weights = s.div(denom, axis=0)

    if fallback == "equal":
        weights = weights.fillna(1.0 / n)
    elif fallback == "zero":
        weights = weights.fillna(0.0)
    else:
        raise ValueError("fallback doit être 'equal' ou 'zero'.")

    return weights


# ============================================================
# 7. SIGNAUX DISCRETS
# ============================================================

def scores_to_signals(
    scores: pd.DataFrame,
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Discrétiser les scores en signaux d'investissement :
        +1  si score >  threshold   (position longue)
         0  si |score| ≤ threshold  (neutre)
        −1  si score < −threshold   (position courte)

    Paramètres
    ----------
    scores : (T, n)
    threshold : float ≥ 0

    Retour
    ------
    signals : (T, n) DataFrame avec valeurs dans {-1, 0, 1}
    """
    scores  = ensure_dataframe(scores, "scores")
    signals = pd.DataFrame(0, index=scores.index, columns=scores.columns)
    signals[scores >  threshold] =  1
    signals[scores < -threshold] = -1
    return signals


# ============================================================
# 8. CIBLE POUR MODÈLE SUPERVISÉ
# ============================================================

def build_target(
    returns_data: pd.DataFrame,
    horizon: int = 1,
    mode: str = "binary"
) -> pd.DataFrame:
    """
    Construire une variable cible future pour l'apprentissage supervisé.

    Modes
    -----
    "binary"     : 1 si r_{t+h} > 0, 0 sinon
    "continuous" : r_{t+h}  (régression)
    "signal"     : signe(r_{t+h}) ∈ {-1, 0, 1}

    Paramètres
    ----------
    returns_data : (T, n)
    horizon : int > 0 — nombre de périodes dans le futur
    mode : str

    Retour
    ------
    target : (T - horizon, n) DataFrame (les dernières lignes sont supprimées)
    """
    returns_data = ensure_dataframe(returns_data, "returns_data")
    if horizon <= 0:
        raise ValueError("horizon doit être strictement positif.")

    future = returns_data.shift(-horizon)

    if mode == "binary":
        target = (future > 0).astype(int)
    elif mode == "continuous":
        target = future
    elif mode == "signal":
        target = np.sign(future)
    else:
        raise ValueError("mode doit être 'binary', 'continuous' ou 'signal'.")

    return target.dropna()


# ============================================================
# 9. ALIGNEMENT ANTI LOOK-AHEAD
# ============================================================

def align_weights_with_returns(
    weights: pd.DataFrame,
    returns_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Décaler les poids d'une période pour éviter la fuite temporelle.

    Principe : les poids calculés avec l'information disponible à t
    ne peuvent être appliqués qu'au rendement r_{t+1}.
    On effectue donc :  w_aligned_t = w_{t-1}

    Les premières lignes (NaN après décalage) et les dernières lignes
    (sans rendement associé) sont supprimées.

    Paramètres
    ----------
    weights : (T, n) — poids calculés à chaque date
    returns_data : (T, n) — rendements réalisés

    Retour
    ------
    (shifted_weights, returns_aligned) : deux DataFrames de même index
    """
    weights      = ensure_dataframe(weights, "weights")
    returns_data = ensure_dataframe(returns_data, "returns_data")

    if list(weights.columns) != list(returns_data.columns):
        raise ValueError("weights et returns_data doivent avoir les mêmes colonnes.")

    shifted = weights.shift(1)                            # décalage d'une période

    common  = shifted.index.intersection(returns_data.index)
    shifted = shifted.loc[common].dropna(how="all")
    ret_aligned = returns_data.loc[shifted.index]

    if len(shifted) == 0:
        raise ValueError("Plus aucune observation après le décalage. Vérifier les données.")

    return shifted, ret_aligned


# ============================================================
# 10. PIPELINE COMPLET
# ============================================================

def build_ml_signals(
    price_data: pd.DataFrame,
    returns_data: pd.DataFrame | None = None,
    window: int = 20,
    method: str = "momentum_volatility",
    long_only: bool = True,
    fallback: str = "equal",
    threshold: float = 0.0,
    align_for_backtest: bool = True
) -> dict:
    """
    Pipeline complet de génération de signaux et de poids dynamiques.

    Étapes
    ------
    1. Calcul des rendements si non fournis
    2. Construction des features statistiques
    3. Calcul des scores selon la méthode choisie
    4. Discrétisation en signaux {-1, 0, 1}
    5. Conversion en poids normalisés
    6. Décalage anti look-ahead (si align_for_backtest=True)

    Méthodes disponibles
    --------------------
    "momentum_volatility" : score = mom / σ̂  (Sharpe glissant)
    "financial_score"     : score composite pondéré (α·mom + β·µ̂ − γ·σ̂)

    Paramètres
    ----------
    price_data : (T, n) DataFrame
    returns_data : (T, n) ou None
    window : int — fenêtre glissante
    method : str
    long_only : bool
    fallback : "equal" ou "zero"
    threshold : float — seuil de discrétisation
    align_for_backtest : bool

    Retour
    ------
    dict avec :
      "features"             : dict[str, DataFrame]
      "scores"               : (T, n) DataFrame
      "signals"              : (T, n) DataFrame  ∈ {-1, 0, 1}
      "weights"              : (T, n) DataFrame  (avant décalage)
      "weights_for_backtest" : (T', n) DataFrame (décalés, si align=True)
      "returns_for_backtest" : (T', n) DataFrame (alignés, si align=True)
    """
    price_data = ensure_dataframe(price_data, "price_data")
    if returns_data is None:
        returns_data = compute_returns(price_data)
    else:
        returns_data = ensure_dataframe(returns_data, "returns_data")

    features = build_features(price_data, returns_data, window)

    if method == "momentum_volatility":
        scores = build_momentum_volatility_scores(price_data, returns_data, window)
    elif method == "financial_score":
        scores = build_asset_scores(price_data, returns_data, window)
    else:
        raise ValueError("method doit être 'momentum_volatility' ou 'financial_score'.")

    signals = scores_to_signals(scores, threshold)
    weights = scores_to_weights(scores, long_only, fallback)

    results = {
        "features": features,
        "scores":   scores,
        "signals":  signals,
        "weights":  weights,
    }

    if align_for_backtest:
        w_bt, r_bt = align_weights_with_returns(weights, returns_data)
        results["weights_for_backtest"] = w_bt
        results["returns_for_backtest"] = r_bt

    return results