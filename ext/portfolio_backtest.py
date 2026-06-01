"""
portfolio_backtest.py
=====================
Évaluation empirique des stratégies de portefeuille.

Ce module contient uniquement les fonctions réutilisables de backtest.
Les graphiques, tableaux finaux et commentaires de résultats doivent rester
dans le notebook .ipynb.

Convention mathématique
-----------------------
On considère n actifs risqués et un actif sans risque.

- r_t in R^n   : rendements simples journaliers des actifs risqués à la date t
- w in R^n     : poids investis dans les actifs risqués (statique ou dynamique)
- w_rf         : poids investi dans l'actif sans risque
- r_f          : taux sans risque ANNUEL
- r_f,daily    : (1 + r_f)^(1/T) - 1   (taux journalier composé exact)

Contrainte budgétaire :
    sum(w) + w_rf = 1,   avec w_i >= 0 et w_rf >= 0 en long-only.

Rendement journalier du portefeuille :
    R_{p,t} = w^T r_t + w_rf * r_f,daily

Valeur cumulée :
    V_t = V_0 * prod_{s=1}^{t} (1 + R_{p,s})

Métriques de performance
------------------------
- Rendement cumulé    : V_T/V_0 - 1
- Rendement annualisé : (V_T/V_0)^{T_ann/T} - 1  (géométrique)
- Volatilité ann.     : sqrt(T_ann) * std(R_{p,t}, ddof=1)
- Sharpe EX-POST      : mean(R_p,t - r_f,daily) / std(R_p,t - r_f,daily) * sqrt(T_ann)
- Max drawdown        : min_t [(V_t - max_{s<=t} V_s) / max_{s<=t} V_s]

Distinction Sharpe ex-ante / ex-post (importante pour le rapport)
-----------------------------------------------------------------
Le Sharpe calculé ici est EX-POST : il est calculé sur les RENDEMENTS RÉALISÉS
du backtest. Il est différent du Sharpe EX-ANTE calculé dans portfolio_math.py
à partir des paramètres estimés (mu, Sigma). Bien distinguer les deux dans le
rapport et dans les tableaux de résultats.

Biais look-ahead
----------------
Pour les stratégies dynamiques, les poids calculés avec l'information jusqu'à t
doivent être appliqués aux rendements de t+1. Ce décalage doit être effectué
en amont via ml_signals.align_weights_with_returns().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ext.portfolio_math import (
    estimate_covariance_matrix,
    portfolio_volatility,
)


# ============================================================
# 1. OUTILS DE VALIDATION
# ============================================================

def ensure_returns_dataframe(
    returns_matrix: pd.DataFrame | np.ndarray,
    name: str = "returns_matrix",
) -> pd.DataFrame:
    """
    Convertir une matrice de rendements en DataFrame et vérifier sa validité.
    """
    if isinstance(returns_matrix, pd.DataFrame):
        R = returns_matrix.copy()
    else:
        R = pd.DataFrame(returns_matrix)

    if R.ndim != 2 or R.shape[0] < 2 or R.shape[1] < 1:
        raise ValueError(
            f"{name} doit être une matrice (T, n) avec T >= 2 et n >= 1."
        )
    if not np.isfinite(R.to_numpy(dtype=float)).all():
        raise ValueError(
            f"{name} contient des NaN ou Inf. Nettoyer les données avant le backtest."
        )

    return R.astype(float)


def risk_free_daily_rate(
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """
    Convertir un taux sans risque annuel en taux journalier composé exact.

        r_f,daily = (1 + r_f)^{1/T} - 1

    Cette formule est préférable à l'approximation r_f/T, qui surestime
    légèrement le taux pour des valeurs élevées de r_f.
    """
    if annualization_factor <= 0:
        raise ValueError("annualization_factor doit être strictement positif.")
    if risk_free_rate <= -1.0:
        raise ValueError("risk_free_rate doit être > -1.")
    return float((1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0)


def validate_static_budget(
    weights: np.ndarray,
    w_rf: float,
    tol: float = 1e-6,
    long_only: bool = True,
) -> np.ndarray:
    """
    Vérifier la contrainte budgétaire d'une stratégie statique.

    Conditions vérifiées :
        sum(w) + w_rf ≈ 1
        w_i >= 0 et w_rf >= 0 si long_only=True
    """
    w = np.asarray(weights, dtype=float)

    if w.ndim != 1:
        raise ValueError("weights doit être un vecteur 1D.")
    if w.size == 0:
        raise ValueError("weights ne doit pas être vide.")
    if not np.isfinite(w).all() or not np.isfinite(w_rf):
        raise ValueError("weights et w_rf doivent être numériques finis.")

    if long_only:
        if (w < -tol).any():
            raise ValueError("Poids négatifs détectés alors que long_only=True.")
        if w_rf < -tol:
            raise ValueError(
                f"w_rf={w_rf:.6f} < 0 : le portefeuille est sur-investi."
            )

    budget = float(w.sum()) + float(w_rf)
    if abs(budget - 1.0) > tol:
        raise ValueError(
            f"Contrainte budgétaire violée : sum(w) + w_rf = {budget:.8f} != 1. "
            "Vérifier que w_rf = 1 - sum(weights)."
        )

    return w


def validate_dynamic_budget(
    weights_over_time: pd.DataFrame,
    tol: float = 1e-6,
    long_only: bool = True,
) -> pd.Series:
    """
    Vérifier les poids dynamiques et calculer w_rf,t = 1 - sum_i w_{t,i}.
    """
    if not np.isfinite(weights_over_time.to_numpy(dtype=float)).all():
        raise ValueError("weights_over_time contient des NaN ou Inf.")

    if long_only and (weights_over_time < -tol).any().any():
        raise ValueError(
            "Certains poids dynamiques sont négatifs alors que long_only=True."
        )

    sum_weights = weights_over_time.sum(axis=1)
    w_rf_series = 1.0 - sum_weights

    if long_only and (w_rf_series < -tol).any():
        raise ValueError(
            "Certains poids dynamiques sont sur-investis : "
            "sum(weights_t) > 1, donc w_rf_t < 0."
        )

    # Supprime les résidus numériques très petits (ex. -1e-15).
    w_rf_series = w_rf_series.where(w_rf_series.abs() > tol, 0.0)
    return w_rf_series.rename("w_rf")


# ============================================================
# 2. RENDEMENTS ET VALEUR DU PORTEFEUILLE
# ============================================================

def compute_portfolio_returns(
    returns_matrix: pd.DataFrame | np.ndarray,
    weights: np.ndarray,
) -> pd.Series:
    """
    Calculer les rendements de la POCHE RISQUÉE uniquement :

        R_{risky,t} = w^T r_t

    L'actif sans risque n'est PAS inclus ici.
    Utiliser run_static_backtest() ou run_dynamic_backtest() pour le
    rendement total intégrant w_rf.
    """
    R = ensure_returns_dataframe(returns_matrix)
    w = np.asarray(weights, dtype=float)

    if R.shape[1] != len(w):
        raise ValueError(
            f"returns_matrix a {R.shape[1]} colonnes mais weights a {len(w)} éléments."
        )

    return pd.Series(
        R.to_numpy() @ w, index=R.index, name="risky_portfolio_returns"
    )


def compute_portfolio_value(
    portfolio_returns: pd.Series | np.ndarray,
    initial_value: float = 1.0,
) -> pd.Series:
    """
    Calculer la valeur cumulée du portefeuille :

        V_t = V_0 * prod_{s=1}^{t} (1 + R_{p,s})

    Paramètres
    ----------
    portfolio_returns : Series
        Rendements journaliers totaux (poche risquée + cash).
    initial_value : float > 0
        Valeur initiale du portefeuille (par exemple 1.0 ou 100.0).
    """
    if not isinstance(portfolio_returns, pd.Series):
        portfolio_returns = pd.Series(
            portfolio_returns, name="portfolio_returns"
        )

    if len(portfolio_returns) < 2:
        raise ValueError(
            "portfolio_returns doit contenir au moins 2 observations."
        )
    if initial_value <= 0:
        raise ValueError("initial_value doit être strictement positif.")
    if not np.isfinite(portfolio_returns.to_numpy(dtype=float)).all():
        raise ValueError("portfolio_returns contient des NaN ou Inf.")
    if (1.0 + portfolio_returns <= 0).any():
        raise ValueError(
            "Au moins un rendement est <= -100 %, "
            "la valeur cumulée devient non positive."
        )

    return pd.Series(
        initial_value * (1.0 + portfolio_returns).cumprod(),
        index=portfolio_returns.index,
        name="portfolio_value",
    )


# ============================================================
# 3. MÉTRIQUES DE PERFORMANCE
# ============================================================

def compute_cumulative_return(portfolio_value: pd.Series) -> float:
    """
    Rendement cumulé total :

        R_cum = V_T / V_0 - 1

    Retourne la performance totale sur la période de backtest.
    """
    if not isinstance(portfolio_value, pd.Series):
        portfolio_value = pd.Series(portfolio_value)
    if len(portfolio_value) < 2:
        raise ValueError("portfolio_value doit contenir au moins 2 valeurs.")

    V0 = float(portfolio_value.iloc[0])
    VT = float(portfolio_value.iloc[-1])
    if V0 <= 0:
        raise ValueError("La valeur initiale doit être strictement positive.")
    return float(VT / V0 - 1.0)


def compute_annualized_return(
    portfolio_value: pd.Series,
    annualization_factor: int = 252,
) -> float:
    """
    Rendement annualisé géométrique :

        r_ann = (V_T / V_0)^{T_ann / T} - 1

    Formule géométrique exacte, adaptée aux rendements composés.
    """
    if not isinstance(portfolio_value, pd.Series):
        portfolio_value = pd.Series(portfolio_value)

    T_obs = len(portfolio_value)
    if T_obs < 2:
        raise ValueError("portfolio_value doit contenir au moins 2 valeurs.")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor doit être strictement positif.")

    V0 = float(portfolio_value.iloc[0])
    VT = float(portfolio_value.iloc[-1])
    if V0 <= 0 or VT <= 0:
        raise ValueError(
            "Les valeurs initiale et finale doivent être strictement positives."
        )

    return float((VT / V0) ** (annualization_factor / T_obs) - 1.0)


def compute_backtest_volatility(
    portfolio_returns: pd.Series | np.ndarray,
    annualization_factor: int = 252,
) -> float:
    """
    Volatilité annualisée RÉALISÉE des rendements du portefeuille :

        sigma_ann = sqrt(T_ann) * std(R_{p,t},  ddof=1)

    Cette volatilité est calculée sur les rendements effectifs du backtest
    (ex-post). Elle peut différer de la volatilité théorique ex-ante
    sqrt(T) * sqrt(w^T Sigma w), qui est basée sur les paramètres estimés.
    Les deux sont complémentaires et doivent apparaître dans le rapport.
    """
    if not isinstance(portfolio_returns, pd.Series):
        portfolio_returns = pd.Series(portfolio_returns)
    if len(portfolio_returns) < 2:
        raise ValueError("Il faut au moins 2 rendements.")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor doit être strictement positif.")

    return float(portfolio_returns.std(ddof=1) * np.sqrt(annualization_factor))


def compute_sharpe_ratio(
    portfolio_returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """
    Ratio de Sharpe annualisé EX-POST :

        SR = mean(R_{p,t} - r_{f,daily}) / std(R_{p,t} - r_{f,daily}) * sqrt(T_ann)

    Toutes les quantités sont calculées sur les rendements RÉALISÉS du backtest.

    ATTENTION : ce Sharpe EX-POST diffère du Sharpe EX-ANTE de portfolio_math.py
    (calculé à partir de mu et Sigma estimés). Dans le rapport :
      - Sharpe ex-ante = mesure de qualité du portefeuille selon le modèle.
      - Sharpe ex-post = performance effectivement réalisée sur l'historique.

    Le taux sans risque annuel est converti en taux journalier composé exact.
    """
    if not isinstance(portfolio_returns, pd.Series):
        portfolio_returns = pd.Series(portfolio_returns)
    if len(portfolio_returns) < 2:
        raise ValueError("Il faut au moins 2 rendements.")

    rf_daily = risk_free_daily_rate(risk_free_rate, annualization_factor)
    excess = portfolio_returns - rf_daily
    sigma = excess.std(ddof=1)

    if np.isclose(sigma, 0.0):
        return 0.0
    return float((excess.mean() / sigma) * np.sqrt(annualization_factor))


def compute_max_drawdown(portfolio_value: pd.Series) -> float:
    """
    Drawdown maximal (Maximum Drawdown, MDD) :

        MDD = min_t [(V_t - max_{s<=t} V_s) / max_{s<=t} V_s]

    Le MDD est négatif ou nul. Un MDD de -0.30 signifie que le portefeuille
    a perdu au maximum 30 % par rapport à son pic historique précédent.
    """
    if not isinstance(portfolio_value, pd.Series):
        portfolio_value = pd.Series(portfolio_value)
    if len(portfolio_value) < 2:
        raise ValueError("portfolio_value doit contenir au moins 2 valeurs.")

    peak = portfolio_value.cummax()
    drawdown = (portfolio_value - peak) / peak
    return float(drawdown.min())


def compute_performance_metrics(
    portfolio_returns: pd.Series,
    portfolio_value: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> dict:
    """
    Calculer toutes les métriques de performance standard.

    Retour
    ------
    dict avec :
      - cumulative_return  : rendement cumulé total
      - annualized_return  : rendement annualisé géométrique (ex-post)
      - volatility         : volatilité annualisée réalisée (ex-post)
      - sharpe_ratio       : Sharpe annualisé EX-POST
      - max_drawdown       : drawdown maximal (valeur négative ou nulle)
    """
    return {
        "cumulative_return": compute_cumulative_return(portfolio_value),
        "annualized_return": compute_annualized_return(
            portfolio_value, annualization_factor
        ),
        "volatility": compute_backtest_volatility(
            portfolio_returns, annualization_factor
        ),
        "sharpe_ratio": compute_sharpe_ratio(
            portfolio_returns,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
        ),
        "max_drawdown": compute_max_drawdown(portfolio_value),
    }


# ============================================================
# 4. BACKTEST STATIQUE
# ============================================================

def run_static_backtest(
    returns_matrix: pd.DataFrame | np.ndarray,
    weights: np.ndarray,
    w_rf: float | None = None,
    initial_value: float = 1.0,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    long_only: bool = True,
) -> dict:
    """
    Backtester une stratégie à poids CONSTANTS.

    Rendement journalier total :
        R_{p,t} = w^T r_t + w_rf * r_f,daily

    Si w_rf=None, il est calculé automatiquement comme résidu budgétaire :
        w_rf = 1 - sum(weights)

    Cela rend la fonction directement compatible avec les poids risqués
    produits par sgd_optimizer.py lorsque la contrainte sum(w) <= 1 est
    utilisée (use_cash=True).

    Retour
    ------
    dict avec :
      - portfolio_returns      : Series des rendements journaliers totaux
      - portfolio_value        : Series de la valeur cumulée
      - risky_returns          : Series des rendements de la poche risquée seule
      - rf_daily               : taux journalier sans risque utilisé
      - weights                : vecteur w utilisé
      - w_rf                   : poids cash utilisé
      - volatility_exante      : volatilité annualisée ex-ante sqrt(T)*sqrt(w^T Sigma w)
                                 (basée sur Sigma estimée sur données de backtest)
      - métriques de performance (voir compute_performance_metrics)
    """
    R = ensure_returns_dataframe(returns_matrix)
    w = np.asarray(weights, dtype=float)

    if R.shape[1] != len(w):
        raise ValueError(
            f"returns_matrix a {R.shape[1]} colonnes mais weights a {len(w)} éléments."
        )

    if w_rf is None:
        w_rf = 1.0 - float(w.sum())

    w = validate_static_budget(w, float(w_rf), long_only=long_only)

    rf_daily = risk_free_daily_rate(risk_free_rate, annualization_factor)
    risky_returns = compute_portfolio_returns(R, w)
    portfolio_returns = (risky_returns + float(w_rf) * rf_daily).rename(
        "portfolio_returns"
    )
    portfolio_value = compute_portfolio_value(portfolio_returns, initial_value)

    metrics = compute_performance_metrics(
        portfolio_returns,
        portfolio_value,
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor,
    )

    # Volatilité analytique ex-ante (formule w^T Sigma w, avec Sigma estimée
    # sur les données du backtest). Étiquetée "exante" pour bien la distinguer
    # de la volatilité réalisée dans metrics["volatility"].
    # NOTE : cette Sigma est estimée sur TOUTES les données du backtest,
    # donc elle utilise de l'information future par rapport à la date initiale.
    # Elle sert uniquement à la comparaison avec la théorie, pas à la mesure
    # de performance prédictive.
    volatility_exante = None
    try:
        Sigma_full = estimate_covariance_matrix(R)
        volatility_exante = (
            portfolio_volatility(w, Sigma_full) * np.sqrt(annualization_factor)
        )
    except Exception:
        pass

    return {
        "portfolio_returns": portfolio_returns,
        "portfolio_value": portfolio_value,
        "risky_returns": risky_returns,
        "rf_daily": rf_daily,
        "weights": w,
        "w_rf": float(w_rf),
        "volatility_exante": volatility_exante,
        **metrics,
    }


# ============================================================
# 5. BACKTEST DYNAMIQUE
# ============================================================

def run_dynamic_backtest(
    returns_matrix: pd.DataFrame | np.ndarray,
    weights_over_time: pd.DataFrame | np.ndarray,
    initial_value: float = 1.0,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    long_only: bool = True,
) -> dict:
    """
    Backtester une stratégie à poids VARIABLES dans le temps.

    Les poids dynamiques sont supposés être DÉJÀ décalés d'une période pour
    éviter le look-ahead bias (via ml_signals.align_weights_with_returns()).

    À chaque date :
        w_rf,t = 1 - sum_i w_{t,i}
        R_{p,t} = sum_i w_{t,i} * r_{t,i} + w_rf,t * r_f,daily

    Retour
    ------
    dict avec :
      - portfolio_returns  : Series des rendements journaliers totaux
      - portfolio_value    : Series de la valeur cumulée
      - risky_returns      : Series des rendements de la poche risquée
      - rf_daily           : taux journalier sans risque utilisé
      - weights_over_time  : DataFrame des poids utilisés
      - w_rf_series        : Series des poids cash à chaque date
      - métriques de performance (voir compute_performance_metrics)
    """
    R = ensure_returns_dataframe(returns_matrix)

    if isinstance(weights_over_time, pd.DataFrame):
        W = weights_over_time.copy().astype(float)
    else:
        W = pd.DataFrame(
            weights_over_time, index=R.index, columns=R.columns
        ).astype(float)

    # Alignement strict sur les dates communes.
    idx = R.index.intersection(W.index)
    R = R.loc[idx]
    W = W.loc[idx]

    if len(idx) < 2:
        raise ValueError(
            "Pas assez d'observations communes entre rendements et poids."
        )
    if R.shape[1] != W.shape[1]:
        raise ValueError(
            "returns_matrix et weights_over_time doivent avoir le même "
            "nombre d'actifs."
        )

    # Vérifie l'ordre des colonnes si les deux ont des noms.
    if list(R.columns) != list(W.columns):
        raise ValueError(
            "Les colonnes de returns_matrix et weights_over_time doivent être "
            "identiques et dans le même ordre."
        )

    w_rf_series = validate_dynamic_budget(W, long_only=long_only)
    rf_daily = risk_free_daily_rate(risk_free_rate, annualization_factor)

    risky_returns = pd.Series(
        (R.to_numpy() * W.to_numpy()).sum(axis=1),
        index=idx,
        name="risky_returns",
    )
    portfolio_returns = (risky_returns + rf_daily * w_rf_series).rename(
        "portfolio_returns"
    )
    portfolio_value = compute_portfolio_value(portfolio_returns, initial_value)

    metrics = compute_performance_metrics(
        portfolio_returns,
        portfolio_value,
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor,
    )

    return {
        "portfolio_returns": portfolio_returns,
        "portfolio_value": portfolio_value,
        "risky_returns": risky_returns,
        "rf_daily": rf_daily,
        "weights_over_time": W,
        "w_rf_series": w_rf_series,
        **metrics,
    }


# ============================================================
# 6. STRATÉGIES DE RÉFÉRENCE (BENCHMARKS)
# ============================================================

def equal_weight_strategy(
    n_assets: int, risky_budget: float = 1.0
) -> np.ndarray:
    """
    Construire un portefeuille équipondéré sur les actifs risqués.

    Par défaut, risky_budget=1.0 :
        w_i = 1/n pour tout i,  w_rf = 0

    Si risky_budget < 1 :
        w_i = risky_budget/n,  w_rf = 1 - risky_budget

    Ce benchmark est la baseline naturelle à comparer avec le SGD optimisé.
    """
    if n_assets <= 0:
        raise ValueError("n_assets doit être un entier strictement positif.")
    if not 0.0 <= risky_budget <= 1.0:
        raise ValueError("risky_budget doit être dans [0, 1].")
    return np.ones(n_assets) * (risky_budget / n_assets)


def cash_strategy(n_assets: int) -> np.ndarray:
    """
    Stratégie 100 % actif sans risque : w = 0, w_rf = 1.

    Utile comme baseline minimale dans le rapport.
    Performance = r_f (taux sans risque annualisé), sans risque de marché.
    """
    if n_assets <= 0:
        raise ValueError("n_assets doit être un entier strictement positif.")
    return np.zeros(n_assets)


# ============================================================
# 7. COMPARAISON DE STRATÉGIES
# ============================================================

def compare_strategies(strategy_results: dict[str, dict]) -> pd.DataFrame:
    """
    Construire un tableau comparatif des métriques principales.

    Pour chaque stratégie, les métriques suivantes sont extraites :
      - cumulative_return   : rendement cumulé total (ex-post)
      - annualized_return   : rendement annualisé géométrique (ex-post)
      - volatility          : volatilité annualisée réalisée (ex-post)
      - volatility_exante   : volatilité analytique (ex-ante, si disponible)
      - sharpe_ratio        : Sharpe annualisé EX-POST (sur rendements réalisés)
      - max_drawdown        : drawdown maximal
      - w_rf / avg_w_rf     : poids cash (statique ou moyen pour dynamique)

    RAPPEL pour le rapport :
      - sharpe_ratio ici est EX-POST (calculé sur les rendements réalisés).
      - La valeur EX-ANTE (calculée sur mu, Sigma) vient de portfolio_sharpe()
        dans portfolio_math.py. Ne pas les confondre dans les tableaux.
    """
    rows = {}
    for name, res in strategy_results.items():
        rows[name] = {
            "cumulative_return":  res.get("cumulative_return"),
            "annualized_return":  res.get("annualized_return"),
            "volatility":         res.get("volatility"),
            "volatility_exante":  res.get("volatility_exante"),
            "sharpe_ratio_expost": res.get("sharpe_ratio"),
            "max_drawdown":       res.get("max_drawdown"),
            "w_rf": res.get("w_rf"),
            "avg_w_rf": (
                float(res["w_rf_series"].mean())
                if isinstance(res.get("w_rf_series"), pd.Series)
                else res.get("w_rf")
            ),
        }
    return pd.DataFrame(rows).T


# ============================================================
# 8. PIPELINE COMPLET
# ============================================================

def run_portfolio_backtest(
    returns_matrix: pd.DataFrame | np.ndarray,
    optimized_weights: np.ndarray | None = None,
    w_rf: float | None = None,
    dynamic_weights: pd.DataFrame | np.ndarray | None = None,
    initial_value: float = 1.0,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    include_cash_baseline: bool = True,
    equal_weight_risky_budget: float = 1.0,
    long_only: bool = True,
) -> dict:
    """
    Pipeline complet de backtest comparant plusieurs stratégies.

    Stratégies incluses
    -------------------
    1. "equal_weight" : baseline équipondérée sur les actifs risqués.
    2. "cash"         : 100 % actif sans risque (si include_cash_baseline=True).
    3. "optimized"    : poids statiques issus de run_sgd() (si fournis).
    4. "dynamic"      : poids dynamiques issus de ml_signals (si fournis).

    Utilisation typique
    -------------------
    sgd_result = run_sgd(R, w0, ...)
    ml_result  = run_ml_sgd(prices, ...)

    backtest = run_portfolio_backtest(
        returns_matrix=R,
        optimized_weights=sgd_result["final_weights"],
        w_rf=sgd_result["w_rf"],
        dynamic_weights=ml_result["weights_for_backtest"],
        risk_free_rate=0.03,
    )
    print(backtest["comparison"])

    Retour
    ------
    dict avec :
      - strategy_results : dict[str, dict] des résultats par stratégie
      - comparison       : DataFrame du tableau comparatif
    """
    R = ensure_returns_dataframe(returns_matrix)
    n_assets = R.shape[1]
    strategy_results: dict[str, dict] = {}

    # 1. Baseline équipondérée.
    w_eq = equal_weight_strategy(
        n_assets, risky_budget=equal_weight_risky_budget
    )
    strategy_results["equal_weight"] = run_static_backtest(
        returns_matrix=R,
        weights=w_eq,
        w_rf=1.0 - float(w_eq.sum()),
        initial_value=initial_value,
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor,
        long_only=long_only,
    )

    # 2. Baseline 100 % cash.
    if include_cash_baseline:
        strategy_results["cash"] = run_static_backtest(
            returns_matrix=R,
            weights=cash_strategy(n_assets),
            w_rf=1.0,
            initial_value=initial_value,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
            long_only=long_only,
        )

    # 3. Stratégie optimisée statique (SGD).
    if optimized_weights is not None:
        strategy_results["optimized"] = run_static_backtest(
            returns_matrix=R,
            weights=np.asarray(optimized_weights, dtype=float),
            w_rf=w_rf,
            initial_value=initial_value,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
            long_only=long_only,
        )

    # 4. Stratégie dynamique (modèle ML).
    if dynamic_weights is not None:
        strategy_results["dynamic"] = run_dynamic_backtest(
            returns_matrix=R,
            weights_over_time=dynamic_weights,
            initial_value=initial_value,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
            long_only=long_only,
        )

    return {
        "strategy_results": strategy_results,
        "comparison": compare_strategies(strategy_results),
    }