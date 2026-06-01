"""
ml_signals.py
=============
Génération de signaux d'investissement et de poids dynamiques par modèle ML.

Rôle du module
--------------
Ce fichier correspond à l'extension Machine Learning du projet.
Il ne réalise pas le backtest et ne produit pas les graphiques du rapport.
Son rôle est uniquement de :

  1. Construire des features financières à partir des prix et rendements.
  2. Transformer ces features en scores ou signaux d'investissement.
  3. Produire des poids dynamiques w_t pour les actifs risqués.
  4. Apprendre une règle linéaire de pondération par SGD (run_ml_sgd).

Convention sur l'actif sans risque
------------------------------------
Les poids retournés concernent UNIQUEMENT les actifs risqués.
La contrainte long-only avec cash impose :

    w_{t,i} >= 0    pour tout i
    sum_i w_{t,i} <= 1

La part résiduelle est l'actif sans risque :

    w_rf,t = 1 - sum_i w_{t,i}

Limitation importante à documenter dans le rapport
---------------------------------------------------
Le modèle ML de ce module est un modèle d'ALLOCATION RELATIVE entre actifs
risqués. Il ne décide PAS du niveau d'exposition globale au risque.

En pratique :
  - Le softmax garantit sum(w_t) = max_risky_weight de façon fixe.
  - Si max_risky_weight = 1.0, le modèle investit toujours 100 % en actifs
    risqués, quelle que soit la période.
  - Le modèle n'arbitre PAS entre actifs risqués et actif sans risque ;
    il répartit un budget risqué fixé à l'avance.

C'est fondamentalement différent du SGD statique de sgd_optimizer.py, qui
peut choisir d'allouer une fraction variable en cash selon les conditions de
marché. Ces deux approches sont complémentaires, pas redondantes.

Pour un modèle ML qui arbitre aussi avec le cash, il faudrait une architecture
différente (ex. réseau produisant n+1 logits incluant l'actif sans risque,
avec softmax global sur n+1 actifs).

Biais look-ahead
----------------
Les features calculées à la date t utilisent l'information disponible jusqu'à t.
Pour appliquer ces poids au rendement réalisé suivant, il faut décaler les
poids d'une période : les poids calculés à t sont appliqués à r_{t+1}.
La fonction align_weights_with_returns() réalise ce décalage obligatoire.

Modèle linéaire appris (run_ml_sgd)
------------------------------------
À chaque date t, pour chaque actif i, on construit :

    score_{t,i}(theta) = alpha * momentum_{t,i}
                       + beta  * mean_return_{t,i}
                       - gamma * volatility_{t,i}

avec theta = (alpha, beta, gamma) in R^3.

Le signe négatif devant gamma est encodé dans la matrice de features :
    F[t, i, :] = [momentum, rolling_mean_return, -rolling_volatility]

de sorte que score = F @ theta, avec gamma >= 0 => pénalisation de la
volatilité.

Les poids sont obtenus par softmax :
    w_t = max_risky_weight * softmax(F_t @ theta)

ce qui garantit w_{t,i} > 0 et sum_i w_{t,i} = max_risky_weight.

Perte utilisée pour apprendre theta (sans actif sans risque) :

    L(theta) = mean_t [ lambda*T*w_t^T Sigma*w_t - T*w_t^T mu ]

où mu et Sigma sont estimés sur l'ensemble des données d'entraînement.
Cette perte apprend une RÈGLE DE PONDÉRATION, pas une prédiction de rendements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ext.sgd_optimizer import compute_learning_rate


# ============================================================
# 1. UTILITAIRES
# ============================================================

def ensure_dataframe(data, name: str = "data") -> pd.DataFrame:
    """Convertir une entrée en DataFrame, avec copie défensive."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame()
    try:
        return pd.DataFrame(data)
    except Exception as exc:
        raise ValueError(
            f"'{name}' ne peut pas être converti en DataFrame."
        ) from exc


def _check_max_risky_weight(max_risky_weight: float) -> None:
    if not 0.0 <= max_risky_weight <= 1.0:
        raise ValueError("max_risky_weight doit appartenir à [0, 1].")


# ============================================================
# 2. CALCUL DES RENDEMENTS SIMPLES
# ============================================================

def compute_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculer les rendements simples :

        r_t = (P_t - P_{t-1}) / P_{t-1}

    La première ligne est NaN (pas de rendement pour le premier prix).
    """
    prices = ensure_dataframe(price_data, "price_data")
    returns = prices.pct_change()
    return returns.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 3. CONSTRUCTION DES FEATURES FINANCIÈRES
# ============================================================

def build_features(
    price_data: pd.DataFrame,
    returns_data: pd.DataFrame | None = None,
    window: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Construire des indicateurs glissants par actif.

    Features retournées
    -------------------
    momentum :
        P_t / P_{t-window} - 1
        Mesure la tendance sur la fenêtre.

    rolling_mean_return :
        Moyenne des rendements sur la fenêtre (avec min_periods=window).

    rolling_volatility :
        Écart-type des rendements sur la fenêtre (correction de Bessel, ddof=1).

    price_vs_ma :
        (P_t - MA_t) / MA_t
        Écart relatif entre le prix courant et sa moyenne mobile.

    risk_adjusted_score :
        rolling_mean_return / rolling_volatility
        Ratio de Sharpe roulant non annualisé (non utilisé dans le modèle
        linéaire, conservé à titre exploratoire).

    Paramètres
    ----------
    window : int > 1
        Fenêtre glissante en jours.
    """
    prices = ensure_dataframe(price_data, "price_data")
    returns = (
        compute_returns(prices)
        if returns_data is None
        else ensure_dataframe(returns_data, "returns_data")
    )

    if window <= 1:
        raise ValueError("window doit être strictement supérieur à 1.")
    if list(prices.columns) != list(returns.columns):
        raise ValueError(
            "price_data et returns_data doivent avoir les mêmes colonnes."
        )

    momentum = prices / prices.shift(window) - 1.0
    rolling_mean_return = returns.rolling(window, min_periods=window).mean()
    rolling_volatility = returns.rolling(window, min_periods=window).std(ddof=1)
    moving_average = prices.rolling(window, min_periods=window).mean()
    price_vs_ma = (prices - moving_average) / moving_average
    risk_adjusted_score = rolling_mean_return / rolling_volatility.replace(
        0.0, np.nan
    )

    features = {
        "momentum": momentum,
        "rolling_mean_return": rolling_mean_return,
        "rolling_volatility": rolling_volatility,
        "price_vs_ma": price_vs_ma,
        "risk_adjusted_score": risk_adjusted_score,
    }

    return {
        key: value.replace([np.inf, -np.inf], np.nan)
        for key, value in features.items()
    }


# ============================================================
# 4. SCORES HEURISTIQUES
# ============================================================

def build_asset_scores(
    price_data: pd.DataFrame,
    returns_data: pd.DataFrame | None = None,
    window: int = 20,
    momentum_weight: float = 0.5,
    return_weight: float = 0.3,
    risk_weight: float = 0.2,
) -> pd.DataFrame:
    """
    Construire un score financier composite :

        score = alpha * momentum + beta * mean_return - gamma * volatility

    où alpha = momentum_weight, beta = return_weight, gamma = risk_weight >= 0.

    Le signe négatif devant gamma signifie que la volatilité est pénalisée :
    un actif plus volatile reçoit un score plus faible, toutes choses égales.

    Ces poids sont heuristiques (fixés a priori). Pour apprendre alpha, beta,
    gamma par optimisation, utiliser run_ml_sgd().
    """
    if risk_weight < 0:
        raise ValueError(
            "risk_weight doit être positif, car il pénalise la volatilité."
        )

    features = build_features(price_data, returns_data, window)
    scores = (
        momentum_weight * features["momentum"]
        + return_weight * features["rolling_mean_return"]
        - risk_weight * features["rolling_volatility"]
    )
    return scores.replace([np.inf, -np.inf], np.nan)


def build_momentum_volatility_scores(
    price_data: pd.DataFrame,
    returns_data: pd.DataFrame | None = None,
    window: int = 20,
) -> pd.DataFrame:
    """
    Construire un score momentum ajusté du risque :

        score = momentum / volatility

    Ce score favorise les actifs ayant une tendance haussière relativement
    à leur volatilité récente. C'est un Sharpe roulant approximatif.
    """
    prices = ensure_dataframe(price_data, "price_data")
    returns = (
        compute_returns(prices)
        if returns_data is None
        else ensure_dataframe(returns_data, "returns_data")
    )

    if window <= 1:
        raise ValueError("window doit être strictement supérieur à 1.")

    momentum = prices / prices.shift(window) - 1.0
    volatility = returns.rolling(window, min_periods=window).std(ddof=1)
    scores = momentum / volatility.replace(0.0, np.nan)
    return scores.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 5. CONVERSION SCORES -> POIDS OU SIGNAUX
# ============================================================

def scores_to_weights(
    scores: pd.DataFrame,
    long_only: bool = True,
    fallback: str = "equal",
    max_risky_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Convertir des scores en poids de portefeuille.

    Cas long-only
    -------------
    On garde uniquement les scores positifs, puis on normalise.
    Si tous les scores d'une ligne sont <= 0 (ou NaN), on applique fallback.

    Cas long-short
    --------------
    On normalise par la somme des valeurs absolues.

    max_risky_weight
    ----------------
    Les poids normalisés sont multipliés par max_risky_weight :

        sum_i w_i = max_risky_weight <= 1
        w_rf = 1 - max_risky_weight  (fixe, indépendant des scores)

    ATTENTION : la poche cash est fixe et ne dépend pas des scores.
    Pour un arbitrage dynamique entre risqué et cash, utiliser le SGD statique.

    Paramètres
    ----------
    fallback : {"equal", "zero"}
        "equal" : répartition équipondérée si scores inexploitables.
        "zero"  : poids nuls (tout en cash) si scores inexploitables.
    """
    _check_max_risky_weight(max_risky_weight)
    sc = ensure_dataframe(scores, "scores")
    n_assets = sc.shape[1]

    if fallback not in {"equal", "zero"}:
        raise ValueError("fallback doit être 'equal' ou 'zero'.")

    if long_only:
        raw = sc.clip(lower=0.0)
        denom = raw.sum(axis=1).replace(0.0, np.nan)
    else:
        raw = sc.copy()
        denom = raw.abs().sum(axis=1).replace(0.0, np.nan)

    weights = raw.div(denom, axis=0)

    if fallback == "equal":
        weights = weights.fillna(1.0 / n_assets)
    else:
        weights = weights.fillna(0.0)

    return weights * max_risky_weight


def scores_to_signals(
    scores: pd.DataFrame, threshold: float = 0.0
) -> pd.DataFrame:
    """
    Transformer des scores en signaux discrets :

        +1  si score > +threshold
         0  si |score| <= threshold
        -1  si score < -threshold
    """
    if threshold < 0:
        raise ValueError("threshold doit être positif ou nul.")

    sc = ensure_dataframe(scores, "scores")
    signals = pd.DataFrame(0, index=sc.index, columns=sc.columns)
    signals[sc > threshold] = 1
    signals[sc < -threshold] = -1
    return signals


# ============================================================
# 6. ALIGNEMENT ANTI LOOK-AHEAD (OBLIGATOIRE)
# ============================================================

def align_weights_with_returns(
    weights: pd.DataFrame,
    returns_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Décaler les poids d'UNE période pour éviter le biais look-ahead.

    Justification :
        Les poids w_t sont calculés avec l'information disponible jusqu'à t.
        Ils ne peuvent donc être appliqués qu'au rendement r_{t+1}.
        Dans une table indexée par date, cela revient à :

            weights_shifted[t+1] = weights[t]

        ce qu'on obtient par weights.shift(1).

    Ce décalage est OBLIGATOIRE avant tout backtest dynamique. Le manquer
    constitue un biais look-ahead qui gonfle artificiellement la performance.

    Paramètres
    ----------
    weights : DataFrame, forme (T, n)
        Poids calculés à chaque date.
    returns_data : DataFrame, forme (T, n)
        Rendements journaliers. Doit avoir les mêmes colonnes.

    Retour
    ------
    (weights_shifted, returns_aligned) : deux DataFrames alignés, sans NaN.
    """
    w = ensure_dataframe(weights, "weights")
    returns = ensure_dataframe(returns_data, "returns_data")

    if list(w.columns) != list(returns.columns):
        raise ValueError(
            "weights et returns_data doivent avoir les mêmes colonnes."
        )

    shifted = w.shift(1)
    common_index = shifted.index.intersection(returns.index)
    shifted = shifted.loc[common_index]
    returns = returns.loc[common_index]

    valid = ~shifted.isna().all(axis=1)
    shifted = shifted.loc[valid]
    returns = returns.loc[valid]

    if len(shifted) == 0:
        raise ValueError(
            "Aucune observation disponible après le décalage anti look-ahead."
        )

    return shifted, returns


# ============================================================
# 7. PIPELINE HEURISTIQUE COMPLET
# ============================================================

def build_ml_signals(
    price_data: pd.DataFrame,
    returns_data: pd.DataFrame | None = None,
    window: int = 20,
    method: str = "momentum_volatility",
    long_only: bool = True,
    fallback: str = "equal",
    threshold: float = 0.0,
    align_for_backtest: bool = True,
    max_risky_weight: float = 1.0,
) -> dict:
    """
    Pipeline complet : features -> scores -> signaux -> poids dynamiques.

    Méthodes disponibles
    --------------------
    "momentum_volatility" :
        score = momentum / volatility  (score par actif, sans paramètre appris)

    "financial_score" :
        score = alpha*momentum + beta*mean_return - gamma*volatility
        avec alpha, beta, gamma heuristiques (non appris).
        Pour des coefficients appris, utiliser run_ml_sgd().

    Retour
    ------
    dict avec :
      - features           : dict des DataFrames de features
      - scores             : DataFrame (T, n) des scores bruts
      - signals            : DataFrame (T, n) des signaux discrets {-1, 0, +1}
      - weights            : DataFrame (T, n) des poids (sans décalage)
      - w_rf               : Series (T,) du poids cash
      - weights_for_backtest : DataFrame décalé d'une période (si align_for_backtest)
      - returns_for_backtest : DataFrame aligné (si align_for_backtest)
      - w_rf_for_backtest    : Series décalée (si align_for_backtest)
    """
    prices = ensure_dataframe(price_data, "price_data")
    returns = (
        compute_returns(prices)
        if returns_data is None
        else ensure_dataframe(returns_data, "returns_data")
    )

    features = build_features(prices, returns, window)

    if method == "momentum_volatility":
        scores = build_momentum_volatility_scores(prices, returns, window)
    elif method == "financial_score":
        scores = build_asset_scores(prices, returns, window)
    else:
        raise ValueError(
            "method doit être 'momentum_volatility' ou 'financial_score'."
        )

    signals = scores_to_signals(scores, threshold)
    weights = scores_to_weights(
        scores,
        long_only=long_only,
        fallback=fallback,
        max_risky_weight=max_risky_weight,
    )

    results = {
        "features": features,
        "scores": scores,
        "signals": signals,
        "weights": weights,
        "w_rf": 1.0 - weights.sum(axis=1),
    }

    if align_for_backtest:
        weights_bt, returns_bt = align_weights_with_returns(
            weights, returns
        )
        results["weights_for_backtest"] = weights_bt
        results["returns_for_backtest"] = returns_bt
        results["w_rf_for_backtest"] = 1.0 - weights_bt.sum(axis=1)

    return results


# ============================================================
# 8. FEATURES POUR LE MODÈLE LINÉAIRE
# ============================================================

def build_linear_feature_tensor(
    features: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, pd.Index, list[str]]:
    """
    Construire le tenseur F utilisé par le modèle linéaire appris.

    Pour chaque date t et actif i :

        F[t, i, :] = [momentum_{t,i}, rolling_mean_{t,i}, -volatility_{t,i}]

    Le signe moins sur la volatilité est intentionnel :
        score = F @ theta = alpha*momentum + beta*mean_return - gamma*volatility
    Si gamma = theta[2] >= 0, la volatilité est bien pénalisée.

    Retour
    ------
    tensor : ndarray, forme (T_valid, n, 3)
    feature_index : pd.Index
        Dates valides (sans NaN sur aucune feature).
    columns : list[str]
        Noms des actifs.
    """
    required = ["momentum", "rolling_mean_return", "rolling_volatility"]
    for key in required:
        if key not in features:
            raise ValueError(f"Feature manquante : {key}")

    momentum = ensure_dataframe(features["momentum"], "momentum")
    mean_ret = ensure_dataframe(
        features["rolling_mean_return"], "rolling_mean_return"
    )
    volatility = ensure_dataframe(
        features["rolling_volatility"], "rolling_volatility"
    )

    common_index = (
        momentum.index.intersection(mean_ret.index).intersection(volatility.index)
    )
    columns = list(momentum.columns)

    momentum = momentum.loc[common_index, columns]
    mean_ret = mean_ret.loc[common_index, columns]
    volatility = volatility.loc[common_index, columns]

    valid = ~(momentum.isna() | mean_ret.isna() | volatility.isna()).any(axis=1)
    momentum = momentum.loc[valid]
    mean_ret = mean_ret.loc[valid]
    volatility = volatility.loc[valid]

    if len(momentum) == 0:
        raise ValueError(
            "Aucune date valide pour construire les features du modèle linéaire."
        )

    tensor = np.stack(
        [
            momentum.to_numpy(dtype=float),
            mean_ret.to_numpy(dtype=float),
            -volatility.to_numpy(dtype=float),  # signe moins => gamma > 0 pénalise
        ],
        axis=2,
    )

    return tensor, momentum.index, columns


# ============================================================
# 9. MODÈLE LINÉAIRE : SCORE -> POIDS
# ============================================================

def linear_model_weights(
    features_t: np.ndarray,
    theta: np.ndarray,
    max_risky_weight: float = 1.0,
) -> np.ndarray:
    """
    Calculer les poids risqués à partir des features d'une date t.

    Architecture :
        score_t = F_t @ theta                (score linéaire par actif)
        w_t = max_risky_weight * softmax(score_t)

    Propriétés garanties par le softmax :
        w_{t,i} > 0  pour tout i
        sum_i w_{t,i} = max_risky_weight

    La stabilité numérique est assurée par le décalage standard du softmax :
        scores <- scores - max(scores)    avant exponentiation.

    Paramètres
    ----------
    features_t : ndarray, forme (n, 3)
        Features [momentum, mean_return, -volatility] pour chaque actif.
    theta : ndarray, forme (3,)
        Paramètres (alpha, beta, gamma).
    max_risky_weight : float dans [0, 1]
        Budget risqué total. Si < 1, une fraction 1-max_risky_weight
        est allouée au cash de façon fixe.
    """
    _check_max_risky_weight(max_risky_weight)

    F = np.asarray(features_t, dtype=float)
    theta = np.asarray(theta, dtype=float)

    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(
            f"features_t doit être de forme (n_assets, 3), reçu {F.shape}."
        )
    if theta.shape != (3,):
        raise ValueError(f"theta doit être de forme (3,), reçu {theta.shape}.")

    scores = F @ theta
    scores = scores - np.max(scores)  # stabilité numérique
    exp_scores = np.exp(scores)
    softmax = exp_scores / exp_scores.sum()
    return max_risky_weight * softmax


def linear_model_gradient(
    features_batch: np.ndarray,
    theta: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    lambda_risk: float = 1.0,
    annualization_factor: int = 252,
    max_risky_weight: float = 1.0,
) -> np.ndarray:
    """
    Calculer le gradient analytique de la perte mean-variance par rapport à theta.

    Perte (sur un mini-batch de taille B) :

        L(theta) = (1/B) * sum_t [ lambda*T*w_t^T Sigma*w_t - T*w_t^T mu ]

    où w_t = max_risky_weight * softmax(F_t @ theta).

    Dérivation par la règle de la chaîne :

        dL/dtheta = (1/B) * sum_t  F_t^T * J_softmax(p_t) * grad_w J_MV(w_t)

    avec :
        grad_w J_MV(w_t) = T * (2*lambda*Sigma*w_t - mu)
        J_softmax(p_t) = c * (diag(p_t) - p_t p_t^T)   (jacobien du softmax)
        c = max_risky_weight

    Remarque : la poche cash (1 - max_risky_weight) est constante et n'apparaît
    pas dans le gradient. Le modèle optimise uniquement la répartition relative
    entre actifs risqués, pas le niveau d'exposition global au risque.

    Paramètres
    ----------
    features_batch : ndarray, forme (B, n, 3)
    theta : ndarray, forme (3,)
    expected_returns : ndarray, forme (n,)
        Rendements moyens journaliers (estimés sur l'ensemble d'entraînement).
    covariance_matrix : ndarray, forme (n, n)
        Covariance journalière (estimée sur l'ensemble d'entraînement).
    """
    _check_max_risky_weight(max_risky_weight)

    X = np.asarray(features_batch, dtype=float)
    theta = np.asarray(theta, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    Sigma = np.asarray(covariance_matrix, dtype=float)

    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError(
            f"features_batch doit être de forme (batch, n_assets, 3), reçu {X.shape}."
        )
    if theta.shape != (3,):
        raise ValueError(f"theta doit être de forme (3,), reçu {theta.shape}.")
    if mu.shape != (X.shape[1],):
        raise ValueError("expected_returns n'a pas la bonne dimension.")
    if Sigma.shape != (X.shape[1], X.shape[1]):
        raise ValueError("covariance_matrix n'a pas la bonne dimension.")

    grad_theta = np.zeros_like(theta, dtype=float)
    batch_size = X.shape[0]
    T_ann = annualization_factor

    for t in range(batch_size):
        F_t = X[t]  # forme (n, 3)

        # Poids via softmax (numériquement stable).
        scores = F_t @ theta
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        p = exp_scores / exp_scores.sum()       # softmax pur, sum = 1
        w = max_risky_weight * p                # poids risqués, sum = max_risky_weight

        # Gradient de J_MV par rapport à w (sans terme cash, w_rf est constant).
        grad_w = T_ann * (2.0 * lambda_risk * Sigma @ w - mu)

        # Jacobien du softmax : dw/dscore = c * (diag(p) - p p^T)
        jacobian = max_risky_weight * (np.diag(p) - np.outer(p, p))

        # Règle de la chaîne : dscore/dtheta = F_t (forme (n, 3))
        # grad_score = jacobian @ grad_w  => forme (n,)
        # grad_theta += F_t^T @ grad_score => forme (3,)
        grad_score = jacobian @ grad_w       # (n,)
        grad_theta += F_t.T @ grad_score    # (3,)

    return grad_theta / batch_size


# ============================================================
# 10. CONTRAINTE SUR THETA
# ============================================================

def project_theta(
    theta: np.ndarray,
    nonnegative_gamma: bool = True,
) -> np.ndarray:
    """
    Projeter theta pour préserver l'interprétation économique.

    Structure de theta = (alpha, beta, gamma) :
        alpha = poids du momentum      (peut être négatif : anti-momentum)
        beta  = poids du rendement moyen (peut être négatif)
        gamma = poids de la volatilité  (>= 0 si nonnegative_gamma=True)

    Pourquoi gamma >= 0 ?
    Si gamma < 0, la volatilité aurait un impact positif sur le score, ce qui
    revient à FAVORISER les actifs volatils. C'est contraire à l'objectif
    mean-variance. La contrainte gamma >= 0 préserve la cohérence économique.

    Paramètres
    ----------
    nonnegative_gamma : bool
        Si False, gamma est libre (pas de contrainte d'interprétabilité).
    """
    theta = np.asarray(theta, dtype=float).copy()
    if theta.shape != (3,):
        raise ValueError("theta doit être de forme (3,).")
    if nonnegative_gamma:
        theta[2] = max(theta[2], 0.0)
    return theta


# ============================================================
# 11. PERTE MEAN-VARIANCE DU MODÈLE LINÉAIRE
# ============================================================

def ml_mean_variance_loss(
    feature_tensor: np.ndarray,
    theta: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    lambda_risk: float = 1.0,
    annualization_factor: int = 252,
    max_risky_weight: float = 1.0,
) -> float:
    """
    Calculer la perte mean-variance moyenne du modèle linéaire sur toutes les dates.

        L(theta) = mean_t [ lambda*T*w_t^T Sigma*w_t - T*w_t^T mu ]

    Note : cette perte N'INCLUT PAS l'actif sans risque. Elle optimise
    uniquement la répartition entre actifs risqués.
    """
    X = np.asarray(feature_tensor, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    Sigma = np.asarray(covariance_matrix, dtype=float)

    losses = []
    for t in range(X.shape[0]):
        w = linear_model_weights(X[t], theta, max_risky_weight=max_risky_weight)
        variance = float(w @ Sigma @ w)
        expected_return = float(w @ mu)
        losses.append(
            lambda_risk * annualization_factor * variance
            - annualization_factor * expected_return
        )

    return float(np.mean(losses))


# ============================================================
# 12. GÉNÉRATION DES POIDS DU MODÈLE
# ============================================================

def generate_linear_model_weights(
    feature_tensor: np.ndarray,
    theta: np.ndarray,
    index: pd.Index,
    columns: list[str],
    max_risky_weight: float = 1.0,
) -> pd.DataFrame:
    """Générer les poids dynamiques du modèle linéaire pour toutes les dates."""
    X = np.asarray(feature_tensor, dtype=float)
    weights = np.vstack([
        linear_model_weights(X[t], theta, max_risky_weight=max_risky_weight)
        for t in range(X.shape[0])
    ])
    return pd.DataFrame(weights, index=index, columns=columns)


# ============================================================
# 13. APPRENTISSAGE DE THETA PAR SGD
# ============================================================

def run_ml_sgd(
    price_data: pd.DataFrame,
    returns_data: pd.DataFrame | None = None,
    window: int = 20,
    theta_init: np.ndarray | None = None,
    n_iterations: int = 500,
    learning_rate: float = 1e-2,
    lr_schedule: str = "constant",
    batch_size: int = 32,
    lambda_risk: float = 1.0,
    annualization_factor: int = 252,
    max_risky_weight: float = 1.0,
    nonnegative_gamma: bool = True,
    random_state: int | None = None,
    verbose: bool = False,
) -> dict:
    """
    Apprendre les paramètres theta = (alpha, beta, gamma) par SGD.

    Le modèle apprend une règle de pondération :
        score = alpha*momentum + beta*mean_return - gamma*volatility
        w_t = max_risky_weight * softmax(score_t)

    Il optimise la répartition RELATIVE entre actifs risqués.
    Il ne décide PAS du niveau d'exposition globale au risque (poche cash fixe).

    Initialisation de theta
    -----------------------
    Si theta_init=None, theta est initialisé à zéro (vecteur nul).
    Ce choix neutre évite de biaiser l'apprentissage vers une solution
    heuristique particulière.

    Schedule du taux d'apprentissage
    ---------------------------------
    Voir sgd_optimizer.py pour la justification des schedules disponibles.
    "constant", "inverse", "sqrt".

    Paramètres
    ----------
    lr_schedule : str
        Schedule du taux d'apprentissage (voir compute_learning_rate).

    Retour
    ------
    dict avec :
      - theta                 : paramètres finaux appris
      - theta_history         : historique de theta
      - loss_history          : historique de la perte
      - weights               : poids dynamiques (sans décalage)
      - w_rf                  : poids cash (Series)
      - weights_for_backtest  : poids décalés d'une période (anti look-ahead)
      - returns_for_backtest  : rendements alignés
      - w_rf_for_backtest     : poids cash décalés
      - features, feature_index, columns, mu, Sigma
    """
    _check_max_risky_weight(max_risky_weight)

    if n_iterations <= 0:
        raise ValueError("n_iterations doit être strictement positif.")
    if learning_rate <= 0:
        raise ValueError("learning_rate doit être strictement positif.")
    if batch_size <= 0:
        raise ValueError("batch_size doit être strictement positif.")
    if lr_schedule not in ("constant", "inverse", "sqrt"):
        raise ValueError("lr_schedule doit être 'constant', 'inverse' ou 'sqrt'.")

    rng = np.random.default_rng(random_state)

    prices = ensure_dataframe(price_data, "price_data")
    returns = (
        compute_returns(prices)
        if returns_data is None
        else ensure_dataframe(returns_data, "returns_data")
    )

    features = build_features(prices, returns, window)
    feature_tensor, feature_index, columns = build_linear_feature_tensor(features)

    returns_valid = returns.loc[feature_index, columns].dropna(how="any")
    common_index = feature_index.intersection(returns_valid.index)
    returns_valid = returns_valid.loc[common_index]

    # Recalage du tenseur de features sur les dates gardées.
    date_position = pd.Series(np.arange(len(feature_index)), index=feature_index)
    positions = date_position.loc[common_index].to_numpy(dtype=int)
    feature_tensor = feature_tensor[positions]

    if len(returns_valid) < 2:
        raise ValueError(
            "Pas assez de données valides pour entraîner le modèle ML."
        )

    R = returns_valid.to_numpy(dtype=float)
    mu = R.mean(axis=0)
    Sigma = np.cov(R, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[float(Sigma)]])

    # Initialisation neutre à zéro (pas de biais vers une heuristique).
    if theta_init is None:
        theta = np.zeros(3, dtype=float)
    else:
        theta = np.asarray(theta_init, dtype=float)
        if theta.shape != (3,):
            raise ValueError("theta_init doit être de forme (3,).")

    theta = project_theta(theta, nonnegative_gamma=nonnegative_gamma)

    n_dates = feature_tensor.shape[0]
    effective_batch_size = min(batch_size, n_dates)

    theta_history = [theta.copy()]
    loss_history = []

    for iteration in range(n_iterations):
        # Schedule du taux d'apprentissage (délégué à sgd_optimizer.compute_learning_rate).
        eta_k = compute_learning_rate(learning_rate, iteration, schedule=lr_schedule)

        batch_idx = rng.choice(n_dates, size=effective_batch_size, replace=False)
        X_batch = feature_tensor[batch_idx]

        grad = linear_model_gradient(
            X_batch,
            theta,
            mu,
            Sigma,
            lambda_risk=lambda_risk,
            annualization_factor=annualization_factor,
            max_risky_weight=max_risky_weight,
        )

        theta = theta - eta_k * grad
        theta = project_theta(theta, nonnegative_gamma=nonnegative_gamma)
        theta_history.append(theta.copy())

        loss = ml_mean_variance_loss(
            feature_tensor,
            theta,
            mu,
            Sigma,
            lambda_risk=lambda_risk,
            annualization_factor=annualization_factor,
            max_risky_weight=max_risky_weight,
        )
        loss_history.append(loss)

        if verbose and (iteration % 50 == 0 or iteration == n_iterations - 1):
            print(
                f"iter={iteration:4d} | loss={loss:.8f} | "
                f"eta={eta_k:.2e} | theta={np.round(theta, 4)}"
            )

    weights = generate_linear_model_weights(
        feature_tensor,
        theta,
        index=common_index,
        columns=columns,
        max_risky_weight=max_risky_weight,
    )

    weights_for_backtest, returns_for_backtest = align_weights_with_returns(
        weights, returns.loc[weights.index, columns]
    )

    return {
        "theta": theta,
        "theta_history": theta_history,
        "loss_history": loss_history,
        "weights": weights,
        "w_rf": 1.0 - weights.sum(axis=1),
        "weights_for_backtest": weights_for_backtest,
        "returns_for_backtest": returns_for_backtest,
        "w_rf_for_backtest": 1.0 - weights_for_backtest.sum(axis=1),
        "features": features,
        "feature_index": common_index,
        "columns": columns,
        "mu": mu,
        "Sigma": Sigma,
        "lr_schedule": lr_schedule,
    }