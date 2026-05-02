"""
portfolio_backtest.py

Description
-----------
Ce module est dédié au backtesting des stratégies de portefeuille.

Son rôle est de :
- simuler l’évolution d’un portefeuille dans le temps
- appliquer des poids fixes ou dynamiques
- calculer les performances obtenues
- comparer plusieurs stratégies d’investissement

Ce module appartient à la partie extension du projet.
Il sera utilisé avec :
- market_data.py
- portfolio_math.py
- sgd_optimizer.py
- éventuellement ml_signals.py
- les notebooks d’analyse
"""


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
from portfolio_math import portfolio_volatility, estimate_covariance_matrix


# ============================================================
# 1. CALCUL DU RENDEMENT DU PORTEFEUILLE A CHAQUE DATE
# ============================================================

def compute_portfolio_returns(returns_matrix, weights):
    """
    Calculer les rendements du portefeuille à partir des rendements
    des actifs et d’un vecteur de poids.

    Paramètres
    ----------
    returns_matrix : DataFrame or array-like
        Matrice des rendements des actifs.
        Chaque ligne correspond à une date,
        chaque colonne correspond à un actif.

    weights : array-like
        Vecteur des poids du portefeuille.

    Retour
    ------
    portfolio_returns : Series or array-like
        Série des rendements du portefeuille.

    Formule
    -------
    r_portfolio(t) = w^T * r(t)

    Utilité
    -------
    Cette fonction permet de transformer des rendements d’actifs
    en rendements de portefeuille.
    """
    if not isinstance(returns_matrix, pd.DataFrame):
        returns_matrix = pd.DataFrame(returns_matrix)
    
    weights = np.array(weights, dtype=float)

    if returns_matrix.shape[1] != len(weights):
        raise ValueError("Le nombre de colonnes de returns_matrix doit correspondre à la longueur de weights.")
    
    portfolio_returns = returns_matrix.values @ weights

    portfolio_returns = pd.Series(
        portfolio_returns,
        index=returns_matrix.index,
        name="Portfolio Returns"
    )

    return portfolio_returns

# ============================================================
# 2. EVOLUTION DE LA VALEUR DU PORTEFEUILLE
# ============================================================

def compute_portfolio_value(portfolio_returns, initial_value=1.0):
    """
    Calculer l’évolution de la valeur cumulée du portefeuille.

    Paramètres
    ----------
    portfolio_returns : Series or array-like
        Série des rendements du portefeuille.
    initial_value : float
        Valeur initiale du portefeuille.

    Retour
    ------
    portfolio_value : Series or array-like
        Valeur du portefeuille au cours du temps.

    Ce qu’il faut faire
    -------------------
    1. Partir d’une valeur initiale
    2. Appliquer les rendements successifs
    3. Construire la trajectoire cumulée du portefeuille

    Utilité
    -------
    Permet de visualiser la performance globale de la stratégie.
    """
    if not isinstance(portfolio_returns, pd.Series):
        portfolio_returns = pd.Series(portfolio_returns)
    
    if initial_value <= 0:
        raise ValueError("initial_value doit être strictement positif.")

    cuml_returns = (1 + portfolio_returns).cumprod()
    portfolio_value =  initial_value * cuml_returns

    portfolio_value.name = "Portfolio Value"

    return portfolio_value


# ============================================================
# 3. RENDEMENT CUMULE
# ============================================================

def compute_cumulative_return(portfolio_value):
    """
    Calculer le rendement cumulé du portefeuille.

    Paramètres
    ----------
    portfolio_value : Series or array-like
        Série de valeur du portefeuille.

    Retour
    ------
    cumulative_return : float
        Rendement cumulé total.

    Formule
    -------
    cumulative_return = final_value / initial_value - 1

    Utilité
    -------
    Mesure simple de la performance totale de la stratégie.
    """
    if not isinstance(portfolio_value, pd.Series):
        portfolio_value = pd.Series(portfolio_value)

    if len(portfolio_value) < 2:
        raise ValueError("portfolio_value doit contenir au moins deux valeurs.")
    
    initial_value = portfolio_value.iloc[0]
    final_value = portfolio_value.iloc[-1]

    if initial_value <= 0:
        raise ValueError("La première valeur de portfolio_value doit être strictement positive.")
    
    cuml_return = final_value / initial_value - 1

    return cuml_return


# ============================================================
# 4. VOLATILITE DU PORTEFEUILLE
# ============================================================

def compute_backtest_volatility(portfolio_returns, annualization_factor=252):
    """
    Calculer la volatilité annualisée du portefeuille.

    Paramètres
    ----------
    portfolio_returns : Series or array-like
        Rendements du portefeuille.
    annualization_factor : int
        Facteur d’annualisation.

    Retour
    ------
    volatility : float
        Volatilité annualisée.

    Utilité
    -------
    Mesure du risque observé pendant le backtest.
    """
    if not isinstance(portfolio_returns, pd.Series):
        portfolio_returns = pd.Series(portfolio_returns)

    if len(portfolio_returns) < 2:
        raise ValueError("portfolio_returns doit contenir au moins deux valeurs.")
    
    vol = portfolio_returns.std()
    annualized_volatility = vol * np.sqrt(annualization_factor)

    return annualized_volatility


# ============================================================
# 5. RATIO DE SHARPE
# ============================================================

def compute_sharpe_ratio(portfolio_returns, risk_free_rate=0.0, annualization_factor=252):
    """
    Calculer le ratio de Sharpe du portefeuille.

    Paramètres
    ----------
    portfolio_returns : Series or array-like
        Rendements du portefeuille.
    risk_free_rate : float
        Taux sans risque.
    annualization_factor : int
        Facteur d’annualisation.

    Retour
    ------
    sharpe_ratio : float
        Ratio de Sharpe observé.

    Idée générale
    -------------
    Le ratio de Sharpe compare le rendement excédentaire
    au risque pris.

    Utilité
    -------
    Permet de comparer la qualité de plusieurs stratégies.
    """
    if not isinstance(portfolio_returns, pd.Series):
        portfolio_returns = pd.Series(portfolio_returns)

    if len(portfolio_returns) < 2:
        raise ValueError("portfolio_returns doit contenir au moins deux valeurs.")
    
    excess_returns = portfolio_returns - risk_free_rate
    mean_excess_return = excess_returns.mean()
    vol = excess_returns.std()

    if np.isclose(vol, 0.0):
        return 0.0
    
    sharpe_ratio = (mean_excess_return / vol) * np.sqrt(annualization_factor)

    return sharpe_ratio

# ============================================================
# 6. DRAWDOWN MAXIMAL
# ============================================================

def compute_max_drawdown(portfolio_value):
    """
    Calculer le drawdown maximal du portefeuille.

    Paramètres
    ----------
    portfolio_value : Series or array-like
        Valeur cumulée du portefeuille.

    Retour
    ------
    max_drawdown : float
        Perte maximale observée depuis un plus haut historique.

    Utilité
    -------
    Mesure importante du risque de perte dans le temps.
    """
    if not isinstance(portfolio_value, pd.Series):
        portfolio_value = pd.Series(portfolio_value)

    if len(portfolio_value) < 2:
        raise ValueError("portfolio_value doit contenir au moins deux valeurs.")

    peak = portfolio_value.cummax()
    drawdown = (portfolio_value - peak) / peak
    max_drawdown = drawdown.min()

    return max_drawdown


# ============================================================
# 7. BACKTEST D’UNE STRATEGIE A POIDS FIXES
# ============================================================

def run_static_backtest(returns_matrix, weights, initial_value=1.0, risk_free_rate=0.0, annualization_factor=252):
    """
    Exécuter le backtest d’une stratégie à poids fixes.

    Paramètres
    ----------
    returns_matrix : DataFrame or array-like
        Rendements des actifs.
    weights : array-like
        Poids constants du portefeuille.
    initial_value : float
        Valeur initiale du portefeuille.
    risk_free_rate : float
        Taux sans risque utilisé pour certaines métriques.

    Retour
    ------
    results : dict
        Dictionnaire contenant par exemple :
        - portfolio_returns
        - portfolio_value
        - cumulative_return
        - volatility
        - sharpe_ratio
        - max_drawdown

    Ce qu’il faut faire
    -------------------
    1. Calculer les rendements du portefeuille
    2. Calculer la valeur cumulée
    3. Calculer les métriques de performance
    4. Retourner les résultats dans une structure simple
    """
    portfolio_returns = compute_portfolio_returns(
        returns_matrix, 
        weights
        )
    
    portfolio_value = compute_portfolio_value(
        portfolio_returns,
        initial_value
        )
    
    cumulative_return = compute_cumulative_return(portfolio_value)
    
    volatility = compute_backtest_volatility(portfolio_returns)
    
    sharpe_ratio = compute_sharpe_ratio(
        portfolio_returns,
        risk_free_rate,
        annualization_factor
    )
    
    max_drawdown = compute_max_drawdown(portfolio_value)

    covariance_matrix = estimate_covariance_matrix(returns_matrix)
    theoretical_volatility = portfolio_volatility(weights, covariance_matrix) * np.sqrt(annualization_factor)

    results = {
        "portfolio_returns": portfolio_returns,
        "portfolio_value": portfolio_value,
        "cumulative_return": cumulative_return,
        "volatility": volatility,
        "theoretical_volatility": theoretical_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }

    return results    


# ============================================================
# 8. BACKTEST D’UNE STRATEGIE A POIDS DYNAMIQUES
# ============================================================

def run_dynamic_backtest(returns_matrix, weights_over_time, initial_value=1.0, risk_free_rate=0.0, annualization_factor=252):
    """
    Exécuter le backtest d’une stratégie à poids variables dans le temps.

    Paramètres
    ----------
    returns_matrix : DataFrame or array-like
        Rendements des actifs.
    weights_over_time : DataFrame or array-like
        Poids du portefeuille à chaque date.
    initial_value : float
        Valeur initiale du portefeuille.
    risk_free_rate : float
        Taux sans risque.

    Retour
    ------
    results : dict
        Structure contenant les performances observées.

    Ce qu’il faut faire
    -------------------
    1. Associer à chaque date le vecteur de poids correspondant
    2. Calculer les rendements du portefeuille date par date
    3. Construire la valeur cumulée
    4. Calculer les métriques finales

    Utilité
    -------
    Cette fonction est utile si vous utilisez des poids optimisés
    ou des signaux dynamiques.
    """
    if not isinstance(returns_matrix, pd.DataFrame):
        returns_matrix = pd.DataFrame(returns_matrix)

    if not isinstance(weights_over_time, pd.DataFrame):
        weights_over_time = pd.DataFrame(
            weights_over_time,
            index=returns_matrix.index,
            columns=returns_matrix.columns)

    common_index = returns_matrix.index.intersection(weights_over_time.index)
    returns_matrix = returns_matrix.loc[common_index]
    weights_over_time = weights_over_time.loc[common_index]

    if returns_matrix.shape[1] != weights_over_time.shape[1]:
        raise ValueError("The number of columns in returns_matrix must match the number of columns in weights_over_time.")
    
    portfolio_returns = pd.Series(
    (returns_matrix * weights_over_time).sum(axis=1),
    index=returns_matrix.index,
    name="Portfolio Returns"
    )

    portfolio_value = compute_portfolio_value(portfolio_returns, initial_value)

    cumulative_return = compute_cumulative_return(portfolio_value)
    volatility = compute_backtest_volatility(portfolio_returns)
    sharpe_ratio = compute_sharpe_ratio(portfolio_returns, risk_free_rate, annualization_factor)
    max_drawdown = compute_max_drawdown(portfolio_value)

    results = {
        "portfolio_returns": portfolio_returns,
        "portfolio_value": portfolio_value,
        "cumulative_return": cumulative_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }

    return results


# ============================================================
# 9. PORTEFEUILLE EQUIPONDERE (BASELINE)
# ============================================================

def equal_weight_strategy(n_assets):
    """
    Construire un portefeuille équipondéré.

    Paramètres
    ----------
    n_assets : int
        Nombre d’actifs.

    Retour
    ------
    weights : array-like
        Vecteur de poids égaux.

    Formule
    -------
    Chaque poids vaut 1 / n_assets.

    Utilité
    -------
    Sert de stratégie de référence simple pour comparaison.
    """
    if n_assets <= 0:
        raise ValueError("n_assets doit être un entier positif.")
    
    weights = np.ones(n_assets) / n_assets

    return weights


# ============================================================
# 10. COMPARAISON DE STRATEGIES
# ============================================================

def compare_strategies(strategy_results):
    """
    Comparer plusieurs stratégies de portefeuille.

    Paramètres
    ----------
    strategy_results : dict
        Dictionnaire contenant les résultats de plusieurs stratégies.

    Retour
    ------
    comparison_table : DataFrame or dict
        Tableau comparatif des métriques principales.

    Ce qu’il faut comparer
    ----------------------
    - rendement cumulé
    - volatilité
    - ratio de Sharpe
    - drawdown maximal

    Utilité
    -------
    Cette fonction permet de résumer clairement les différences
    entre les stratégies testées.
    """
    comparison = {}

    for strategy_name, results in strategy_results.items():
        comparison[strategy_name] = {
            "cumulative_return": results["cumulative_return"],
            "volatility": results["volatility"],
            "theoretical_volatility": results.get("theoretical_volatility", None),
            "sharpe_ratio": results["sharpe_ratio"],
            "max_drawdown": results["max_drawdown"]
        }
    
    comparison_table = pd.DataFrame(comparison).T

    return comparison_table


# ============================================================
# 11. PIPELINE COMPLET DE BACKTEST
# ============================================================

def run_portfolio_backtest(
    returns_matrix,
    optimized_weights=None,
    dynamic_weights=None,
    initial_value=1.0,
    risk_free_rate=0.0,
    annualization_factor=252
):
    """
    Lancer un backtest complet et comparer plusieurs stratégies.

    Paramètres
    ----------
    returns_matrix : DataFrame or array-like
        Rendements des actifs.
    optimized_weights : array-like or None
        Poids optimisés obtenus par exemple avec sgd_optimizer.py
    dynamic_weights : DataFrame or array-like or None
        Poids variables dans le temps.
    initial_value : float
        Valeur initiale du portefeuille.
    risk_free_rate : float
        Taux sans risque.
    annualization_factor : int
        Facteur d'annualisation.

    Retour
    ------
    backtest_results : dict
        Dictionnaire global contenant :
        - résultats de la stratégie équipondérée
        - résultats de la stratégie optimisée
        - résultats éventuels de la stratégie dynamique
        - tableau de comparaison

    Ce qu’il faut faire
    -------------------
    1. Construire la stratégie baseline équipondérée
    2. Exécuter son backtest
    3. Si des poids optimisés sont fournis, exécuter leur backtest
    4. Si des poids dynamiques sont fournis, exécuter leur backtest
    5. Comparer toutes les stratégies
    6. Retourner les résultats complets

    Objectif
    --------
    Fournir une interface unique pour l’évaluation des stratégies.
    """

    if not isinstance(returns_matrix, pd.DataFrame):
        returns_matrix = pd.DataFrame(returns_matrix)

    n_assets = returns_matrix.shape[1]

    strategy_results = {}

    # 1. Stratégie équipondérée
    equal_weights = equal_weight_strategy(n_assets)

    results_equal = run_static_backtest(
        returns_matrix=returns_matrix,
        weights=equal_weights,
        initial_value=initial_value,
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor
    )

    strategy_results["equal_weight"] = results_equal

    # 2. Stratégie optimisée statique
    if optimized_weights is not None:
        results_optimized = run_static_backtest(
            returns_matrix=returns_matrix,
            weights=optimized_weights,
            initial_value=initial_value,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor
        )

        strategy_results["optimized"] = results_optimized

    # 3. Stratégie dynamique
    if dynamic_weights is not None:
        results_dynamic = run_dynamic_backtest(
            returns_matrix=returns_matrix,
            weights_over_time=dynamic_weights,
            initial_value=initial_value,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor
        )

        strategy_results["dynamic"] = results_dynamic

    # 4. Tableau comparatif
    comparison = compare_strategies(strategy_results)

    backtest_results = {
        "strategy_results": strategy_results,
        "comparison": comparison
    }

    return backtest_results

# ============================================================
# 12. NOTES D’UTILISATION
# ============================================================

"""
Utilisation typique
-------------------
1. Récupérer les rendements de plusieurs actifs
2. Construire un portefeuille de référence équipondéré
3. Construire un portefeuille optimisé via sgd_optimizer.py
4. Backtester les deux stratégies
5. Comparer leurs performances

Exemple logique
---------------
returns_matrix -> weights -> portfolio_returns -> portfolio_value -> performance metrics

Remarque importante
-------------------
Ce module évalue les stratégies, mais ne calcule pas lui-même
les poids optimaux. Cette étape doit être réalisée en amont.

Séparation des rôles
--------------------
- market_data.py : fournit les données de marché
- portfolio_math.py : définit les quantités mathématiques
- sgd_optimizer.py : optimise les poids
- ml_signals.py : génère éventuellement des poids dynamiques
- portfolio_backtest.py : mesure la performance des stratégies
"""