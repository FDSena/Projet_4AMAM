"""
portfolio_backtest.py
=====================
Évaluation empirique des stratégies de portefeuille.

Fondements théoriques
---------------------
Le backtest simule la performance historique d'une stratégie.
On suppose des rendements simples r_t pour chaque actif.

Rendement du portefeuille à la date t :
    R_p,t = wᵀ r_t

Valeur cumulée (avec valeur initiale V₀) :
    V_t = V₀ · ∏_{s=1}^{t} (1 + R_p,s)

Métriques de performance
------------------------
Rendement cumulé    : V_T / V₀ − 1
Rendement annualisé : (V_T / V₀)^(T_ann/T) − 1
Volatilité ann.     : √T_ann · std(R_p,t)
Ratio de Sharpe     : (µ_exc · T_ann) / (σ · √T_ann)
                    = µ_exc · √T_ann / σ
Max drawdown        : min_t (V_t − max_{s≤t} V_s) / max_{s≤t} V_s

Biais look-ahead (stratégies dynamiques)
----------------------------------------
Les poids calculés à la date t utilisent de l'information disponible
jusqu'à t. Pour éviter la fuite temporelle, on décale les poids
d'une période : w_{t+1} est appliqué au rendement r_{t+1}.
Ce décalage est géré dans ml_signals.py (align_weights_with_returns).
"""

import numpy as np
import pandas as pd
from ext.portfolio_math import (
    portfolio_volatility,
    estimate_covariance_matrix,
)


# ============================================================
# 1. RENDEMENTS DU PORTEFEUILLE
# ============================================================

def compute_portfolio_returns(
    returns_matrix: pd.DataFrame | np.ndarray,
    weights: np.ndarray
) -> pd.Series:
    """
    Calculer R_p,t = wᵀ r_t pour chaque date t.

    Paramètres
    ----------
    returns_matrix : (T, n) DataFrame ou array
    weights : (n,)

    Retour
    ------
    portfolio_returns : (T,) Series
    """
    if not isinstance(returns_matrix, pd.DataFrame):
        returns_matrix = pd.DataFrame(returns_matrix)

    w = np.asarray(weights, dtype=float)
    if returns_matrix.shape[1] != len(w):
        raise ValueError(
            f"returns_matrix a {returns_matrix.shape[1]} colonnes "
            f"mais weights a {len(w)} éléments."
        )

    return pd.Series(
        returns_matrix.values @ w,
        index=returns_matrix.index,
        name="portfolio_returns"
    )


# ============================================================
# 2. VALEUR CUMULÉE
# ============================================================

def compute_portfolio_value(
    portfolio_returns: pd.Series | np.ndarray,
    initial_value: float = 1.0
) -> pd.Series:
    """
    Calculer V_t = V₀ · ∏_{s=1}^{t} (1 + R_p,s).

    Paramètres
    ----------
    portfolio_returns : (T,) Series
    initial_value : float > 0

    Retour
    ------
    portfolio_value : (T,) Series
    """
    if not isinstance(portfolio_returns, pd.Series):
        portfolio_returns = pd.Series(portfolio_returns)
    if initial_value <= 0:
        raise ValueError("initial_value doit être strictement positif.")

    return pd.Series(
        initial_value * (1.0 + portfolio_returns).cumprod(),
        index=portfolio_returns.index,
        name="portfolio_value"
    )


# ============================================================
# 3. RENDEMENT CUMULÉ
# ============================================================

def compute_cumulative_return(portfolio_value: pd.Series) -> float:
    """
    Rendement cumulé : V_T / V₀ − 1.
    """
    if not isinstance(portfolio_value, pd.Series):
        portfolio_value = pd.Series(portfolio_value)
    if len(portfolio_value) < 2:
        raise ValueError("portfolio_value doit contenir au moins 2 valeurs.")
    V0 = portfolio_value.iloc[0]
    if V0 <= 0:
        raise ValueError("La valeur initiale doit être strictement positive.")
    return float(portfolio_value.iloc[-1] / V0 - 1.0)


# ============================================================
# 4. RENDEMENT ANNUALISÉ
# ============================================================

def compute_annualized_return(
    portfolio_value: pd.Series,
    annualization_factor: int = 252
) -> float:
    """
    Rendement annualisé géométrique :

        r_ann = (V_T / V₀)^(T_ann / T) − 1

    où T = nombre d'observations et T_ann = annualization_factor.
    """
    if not isinstance(portfolio_value, pd.Series):
        portfolio_value = pd.Series(portfolio_value)
    T = len(portfolio_value)
    if T < 2:
        raise ValueError("portfolio_value doit contenir au moins 2 valeurs.")
    V0 = portfolio_value.iloc[0]
    VT = portfolio_value.iloc[-1]
    if V0 <= 0:
        raise ValueError("La valeur initiale doit être strictement positive.")
    return float((VT / V0) ** (annualization_factor / T) - 1.0)


# ============================================================
# 5. VOLATILITÉ RÉALISÉE
# ============================================================

def compute_backtest_volatility(
    portfolio_returns: pd.Series,
    annualization_factor: int = 252
) -> float:
    """
    Volatilité annualisée réalisée :

        σ_ann = √T_ann · std(R_p,t)

    L'écart-type est calculé avec le correcteur de Bessel (ddof=1).

    Paramètres
    ----------
    portfolio_returns : (T,)
    annualization_factor : int

    Retour
    ------
    volatility : float
    """
    if not isinstance(portfolio_returns, pd.Series):
        portfolio_returns = pd.Series(portfolio_returns)
    if len(portfolio_returns) < 2:
        raise ValueError("Il faut au moins 2 rendements.")
    return float(portfolio_returns.std(ddof=1) * np.sqrt(annualization_factor))


# ============================================================
# 6. RATIO DE SHARPE
# ============================================================

def compute_sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Ratio de Sharpe annualisé ex-post :

        SR = (µ_exc · T_ann) / (σ · √T_ann)
           = (µ_exc / σ) · √T_ann

    où µ_exc = mean(R_p,t − r_f/T_ann) est le rendement excédentaire
    journalier moyen et σ son écart-type (Bessel).

    Paramètres
    ----------
    portfolio_returns : (T,)
    risk_free_rate : float — taux annuel
    annualization_factor : int

    Retour
    ------
    sharpe : float  (0.0 si σ ≈ 0)
    """
    if not isinstance(portfolio_returns, pd.Series):
        portfolio_returns = pd.Series(portfolio_returns)
    if len(portfolio_returns) < 2:
        raise ValueError("Il faut au moins 2 rendements.")

    rf_daily = risk_free_rate / annualization_factor
    excess   = portfolio_returns - rf_daily
    mu_exc   = excess.mean()
    sigma    = excess.std(ddof=1)

    if np.isclose(sigma, 0.0):
        return 0.0
    return float((mu_exc / sigma) * np.sqrt(annualization_factor))


# ============================================================
# 7. MAX DRAWDOWN
# ============================================================

def compute_max_drawdown(portfolio_value: pd.Series) -> float:
    """
    Drawdown maximal (valeur négative) :

        MDD = min_t [(V_t − max_{s≤t} V_s) / max_{s≤t} V_s]

    Mesure la perte maximale subie depuis un pic historique.
    MDD ∈ [−1, 0],  MDD = 0 si la valeur est toujours croissante.

    Paramètres
    ----------
    portfolio_value : (T,)

    Retour
    ------
    max_drawdown : float ≤ 0
    """
    if not isinstance(portfolio_value, pd.Series):
        portfolio_value = pd.Series(portfolio_value)
    if len(portfolio_value) < 2:
        raise ValueError("portfolio_value doit contenir au moins 2 valeurs.")

    peak     = portfolio_value.cummax()
    drawdown = (portfolio_value - peak) / peak
    return float(drawdown.min())


# ============================================================
# 8. BACKTEST STRATÉGIE STATIQUE
# ============================================================

def run_static_backtest(
    returns_matrix: pd.DataFrame | np.ndarray,
    weights: np.ndarray,
    w_rf: float = 0.0,
    initial_value: float = 1.0,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> dict:
    """
    Backtester une stratégie à poids constants.

    Le rendement journalier de l'actif sans risque est :
        r_f_daily = (risk_free_rate / T_ann) · w_rf

    Il est ajouté au rendement quotidien du portefeuille risqué.

    Paramètres
    ----------
    returns_matrix : (T, n)
    weights : (n,) — poids sur les actifs risqués (somme ≤ 1)
    w_rf : float — part investie en actif sans risque
    initial_value : float
    risk_free_rate : float — annuel
    annualization_factor : int

    Retour
    ------
    dict avec portfolio_returns, portfolio_value, et toutes les métriques.
    """
    if not isinstance(returns_matrix, pd.DataFrame):
        returns_matrix = pd.DataFrame(returns_matrix)

    w = np.asarray(weights, dtype=float)

    rf_daily       = (risk_free_rate / annualization_factor) * w_rf
    port_returns   = compute_portfolio_returns(returns_matrix, w) + rf_daily
    port_value     = compute_portfolio_value(port_returns, initial_value)

    cum_return     = compute_cumulative_return(port_value)
    ann_return     = compute_annualized_return(port_value, annualization_factor)
    volatility     = compute_backtest_volatility(port_returns, annualization_factor)
    sharpe         = compute_sharpe_ratio(port_returns, risk_free_rate, annualization_factor)
    max_dd         = compute_max_drawdown(port_value)

    # Volatilité théorique ex-ante (sur les actifs risqués)
    try:
        Sigma        = estimate_covariance_matrix(returns_matrix)
        theor_vol    = portfolio_volatility(w, Sigma) * np.sqrt(annualization_factor)
    except Exception:
        theor_vol    = None

    return {
        "portfolio_returns":      port_returns,
        "portfolio_value":        port_value,
        "cumulative_return":      cum_return,
        "annualized_return":      ann_return,
        "volatility":             volatility,
        "theoretical_volatility": theor_vol,
        "sharpe_ratio":           sharpe,
        "max_drawdown":           max_dd,
        "w_rf":                   w_rf,
    }


# ============================================================
# 9. BACKTEST STRATÉGIE DYNAMIQUE
# ============================================================

def run_dynamic_backtest(
    returns_matrix: pd.DataFrame | np.ndarray,
    weights_over_time: pd.DataFrame | np.ndarray,
    initial_value: float = 1.0,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> dict:
    """
    Backtester une stratégie à poids variables dans le temps.

    Attention au biais look-ahead : les poids de la date t doivent
    avoir été calculés sans utiliser le rendement r_t.
    Ce décalage doit être effectué en amont (ml_signals.align_weights_with_returns).

    Paramètres
    ----------
    returns_matrix : (T, n) — rendements réalisés
    weights_over_time : (T, n) — poids pour chaque date
    initial_value : float
    risk_free_rate : float — annuel
    annualization_factor : int

    Retour
    ------
    dict avec toutes les métriques.
    """
    if not isinstance(returns_matrix, pd.DataFrame):
        returns_matrix = pd.DataFrame(returns_matrix)
    if not isinstance(weights_over_time, pd.DataFrame):
        weights_over_time = pd.DataFrame(
            weights_over_time,
            index=returns_matrix.index,
            columns=returns_matrix.columns
        )

    # Alignement sur l'index commun
    idx               = returns_matrix.index.intersection(weights_over_time.index)
    returns_matrix    = returns_matrix.loc[idx]
    weights_over_time = weights_over_time.loc[idx]

    if returns_matrix.shape[1] != weights_over_time.shape[1]:
        raise ValueError("returns_matrix et weights_over_time n'ont pas le même nombre d'actifs.")

    port_returns = pd.Series(
        (returns_matrix.values * weights_over_time.values).sum(axis=1),
        index=idx,
        name="portfolio_returns"
    )

    port_value   = compute_portfolio_value(port_returns, initial_value)
    cum_return   = compute_cumulative_return(port_value)
    ann_return   = compute_annualized_return(port_value, annualization_factor)
    volatility   = compute_backtest_volatility(port_returns, annualization_factor)
    sharpe       = compute_sharpe_ratio(port_returns, risk_free_rate, annualization_factor)
    max_dd       = compute_max_drawdown(port_value)

    return {
        "portfolio_returns": port_returns,
        "portfolio_value":   port_value,
        "cumulative_return": cum_return,
        "annualized_return": ann_return,
        "volatility":        volatility,
        "sharpe_ratio":      sharpe,
        "max_drawdown":      max_dd,
    }


# ============================================================
# 10. STRATÉGIE ÉQUIPONDÉRÉE (BASELINE)
# ============================================================

def equal_weight_strategy(n_assets: int) -> np.ndarray:
    """
    Construire le portefeuille équipondéré : w_i = 1/n.

    Il minimise la variance dans le cas isotrope (Σ = σ²I)
    et sert de baseline universelle.
    """
    if n_assets <= 0:
        raise ValueError("n_assets doit être un entier strictement positif.")
    return np.ones(n_assets) / n_assets


# ============================================================
# 11. COMPARAISON DE STRATÉGIES
# ============================================================

def compare_strategies(strategy_results: dict) -> pd.DataFrame:
    """
    Construire un tableau comparatif des métriques principales.

    Paramètres
    ----------
    strategy_results : dict[str, dict]
        Clé = nom de la stratégie, valeur = résultats de backtest.

    Retour
    ------
    comparison : DataFrame, lignes = stratégies, colonnes = métriques.
    """
    rows = {}
    for name, res in strategy_results.items():
        rows[name] = {
            "cumulative_return":      res.get("cumulative_return"),
            "annualized_return":      res.get("annualized_return"),
            "volatility":             res.get("volatility"),
            "theoretical_volatility": res.get("theoretical_volatility"),
            "sharpe_ratio":           res.get("sharpe_ratio"),
            "max_drawdown":           res.get("max_drawdown"),
        }
    return pd.DataFrame(rows).T


# ============================================================
# 12. PIPELINE COMPLET
# ============================================================

def run_portfolio_backtest(
    returns_matrix: pd.DataFrame | np.ndarray,
    optimized_weights: np.ndarray | None = None,
    w_rf: float = 0.0,
    dynamic_weights: pd.DataFrame | np.ndarray | None = None,
    initial_value: float = 1.0,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> dict:
    """
    Pipeline complet : backtest de toutes les stratégies disponibles
    et tableau de comparaison.

    Stratégies incluses
    -------------------
    1. equal_weight  : poids égaux  (toujours inclus, sert de baseline)
    2. optimized     : poids issus de run_sgd (si fournis)
    3. dynamic       : poids dynamiques issus de ml_signals (si fournis)

    Paramètres
    ----------
    returns_matrix : (T, n)
    optimized_weights : (n,) or None
    w_rf : float — part de l'actif sans risque dans la stratégie optimisée
    dynamic_weights : (T, n) or None
    initial_value : float
    risk_free_rate : float — annuel
    annualization_factor : int

    Retour
    ------
    dict avec :
      "strategy_results" : dict[str, dict]  — résultats par stratégie
      "comparison"       : DataFrame        — tableau comparatif
    """
    if not isinstance(returns_matrix, pd.DataFrame):
        returns_matrix = pd.DataFrame(returns_matrix)

    n_assets         = returns_matrix.shape[1]
    strategy_results = {}

    # 1. Baseline équipondérée
    strategy_results["equal_weight"] = run_static_backtest(
        returns_matrix=returns_matrix,
        weights=equal_weight_strategy(n_assets),
        initial_value=initial_value,
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor
    )

    # 2. Stratégie optimisée (poids fixes issus du SGD)
    if optimized_weights is not None:
        strategy_results["optimized"] = run_static_backtest(
            returns_matrix=returns_matrix,
            weights=np.asarray(optimized_weights, dtype=float),
            w_rf=w_rf,
            initial_value=initial_value,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor
        )

    # 3. Stratégie dynamique (poids variables issus de ml_signals)
    if dynamic_weights is not None:
        strategy_results["dynamic"] = run_dynamic_backtest(
            returns_matrix=returns_matrix,
            weights_over_time=dynamic_weights,
            initial_value=initial_value,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor
        )

    return {
        "strategy_results": strategy_results,
        "comparison":       compare_strategies(strategy_results),
    }