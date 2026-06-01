"""
backtester.py

Description
-----------
Ce module est dédié à l’évaluation du modèle de Cox-Ross-Rubinstein (CRR)
sur des données réelles ou sur une série de dates.

Son rôle est de :
- appliquer le modèle de pricing sur plusieurs observations
- comparer les prix prédits à des prix de référence
- calculer des métriques d’erreur
- analyser la stabilité du modèle dans le temps

Ce module est utilisé principalement dans :
- les notebooks d’analyse
- l’évaluation empirique du projet
"""


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# 1. COMPARAISON ENTRE PRIX PREVU ET PRIX OBSERVE
# ============================================================

def compare_prices(predicted_prices, observed_prices):
    """
    Comparer une série de prix prédits à une série de prix observés.

    Paramètres
    ----------
    predicted_prices : array-like
        Prix calculés par le modèle CRR.

    observed_prices : array-like
        Prix observés ou prix de référence.

    Retour
    ------
    comparison_df : DataFrame ou structure équivalente
        Tableau contenant par exemple :
        - prix prédit
        - prix observé
        - erreur absolue
        - erreur relative

    Ce qu’il faut faire
    -------------------
    1. Vérifier que les deux séries ont la même taille
    2. Construire une structure de comparaison claire
    3. Calculer les écarts entre prédiction et observation

    Objectif
    --------
    Permettre une lecture simple des résultats avant même de calculer
    des métriques globales.
    """
    predicted = np.array(predicted_prices)
    observed = np.array(observed_prices)
    if len(predicted) != len(observed):
        raise ValueError(f"Tailles incompatibles entre prix prédits {len(predicted)} et prix observés {len(observed)}.")
    return pd.DataFrame({
        'predicted_price': predicted,
        'observed_price': observed,
        'absolute_error': np.abs(predicted - observed),
        'relative_error': np.abs(predicted - observed) / np.where(observed != 0, observed, 1)  # éviter division par zéro
    })
    pass


# ============================================================
# 2. CALCUL DES ERREURS ABSOLUES
# ============================================================

def compute_absolute_errors(predicted_prices, observed_prices):
    """
    Calculer les erreurs absolues entre prix prédits et prix observés.

    Paramètres
    ----------
    predicted_prices : array-like
        Prix calculés par le modèle.

    observed_prices : array-like
        Prix observés.

    Retour
    ------
    abs_errors : array-like
        Série des erreurs absolues.

    Formule
    -------
    |predicted - observed|

    Utilité
    -------
    Base pour les métriques comme la MAE.
    """
    predicted = np.array(predicted_prices)
    observed = np.array(observed_prices)
    if len(predicted) != len(observed):
        raise ValueError(f"Tailles incompatibles entre prix prédits {len(predicted)} et prix observés {len(observed)}.")
    return np.abs(predicted - observed)
    pass


# ============================================================
# 3. MAE
# ============================================================

def compute_mae(predicted_prices, observed_prices):
    """
    Calculer la Mean Absolute Error (MAE).

    Paramètres
    ----------
    predicted_prices : array-like
    observed_prices : array-like

    Retour
    ------
    mae : float

    Formule
    -------
    MAE = moyenne des erreurs absolues

    Utilité
    -------
    Fournit une mesure simple de l’écart moyen entre modèle et référence.
    """
    return np.mean(compute_absolute_errors(predicted_prices, observed_prices))
    pass


# ============================================================
# 4. RMSE
# ============================================================

def compute_rmse(predicted_prices, observed_prices):
    """
    Calculer la Root Mean Squared Error (RMSE).

    Paramètres
    ----------
    predicted_prices : array-like
    observed_prices : array-like

    Retour
    ------
    rmse : float

    Formule
    -------
    RMSE = racine carrée de la moyenne des carrés des erreurs

    Utilité
    -------
    Cette métrique pénalise davantage les grosses erreurs.
    """
    error = compute_absolute_errors(predicted_prices, observed_prices)
    return np.sqrt(np.mean(error ** 2))
    pass


# ============================================================
# 5. BACKTEST SUR PLUSIEURS DATES
# ============================================================

def run_backtest(data, option, pricing_function, calibration_function, n_steps, risk_free_rate=None, window=252):
    """
    Exécuter un backtest du modèle sur plusieurs dates.

    Paramètres
    ----------
    window : int
        Nombre de jours historiques utilisés pour calibrer la volatilité à chaque date.

    data : structure de données
        Données de marché nécessaires au backtest.
        Cela peut contenir :
        - les prix de l’actif sous-jacent
        - éventuellement des prix d’options observés
        - les rendements historiques

    option : Option
        Objet option défini dans option.py

    pricing_function : callable
        Fonction de pricing du modèle CRR.

    calibration_function : callable
        Fonction de calibration fournissant sigma, r, dt, u, d, p_star.

    n_steps : int
        Nombre d’étapes du modèle binomial.

    risk_free_rate : float or None
        Taux sans risque éventuellement fixé manuellement.

    Retour
    ------
    results : DataFrame ou dictionnaire
        Structure contenant pour chaque date :
        - la date
        - le prix du sous-jacent
        - le prix prédit
        - le prix observé ou de référence
        - les erreurs
        - éventuellement les paramètres calibrés

    Ce qu’il faut faire
    -------------------
    1. Parcourir les différentes dates de test
    2. À chaque date :
        - récupérer les données nécessaires
        - calibrer les paramètres du modèle
        - calculer le prix théorique de l’option
    3. Stocker tous les résultats dans une structure claire
    4. Retourner l’ensemble des résultats

    Remarque
    --------
    Le mot “backtest” signifie ici qu’on répète l’évaluation du modèle
    sur plusieurs dates pour voir s’il reste cohérent dans le temps.
    """

    
    data = data.copy()
    data.index = pd.to_datetime(data.index)  if not isinstance(data.index, pd.DatetimeIndex) else data.index
    data = data.sort_index()

    results = []
    
    for i in range(window, len(data)):
        date = data.index[i]
        S = data["Close"].iloc[i]

        # fenetre glissante pour la calibration
        log_returns_window = data["log_return"].iloc[i - window:i].dropna()

        try :
            params = calibration_function(
                log_returns_window,
                maturity=option.maturity,
                n_steps=n_steps,
                risk_free_rate=risk_free_rate)
            price, _ = pricing_function(
                S0=S,
                option=option,
                u=params["u"],
                d=params["d"],
                r=params["r"],
                dt=params["dt"],
                p_star=params["p_star"],
                american=False
            )
        except Exception as e:
            price = np.nan
            params = {}
        
        results.append({
            "date": date,
            "S":S,
            "prix_prédit": price,
            "sigma" : params.get("sigma"),
            "r" : params.get("r"),
            "dt" : params.get("dt"),
            "u" : params.get("u"),
            "d" : params.get("d"),
            "p_star" : params.get("p_star")
        })
    
    return pd.DataFrame(results)
    pass


# ============================================================
# 6. RESUME DES METRIQUES
# ============================================================

def summarize_backtest_results(results):
    """
    Résumer les résultats du backtest avec quelques métriques globales.

    Paramètres
    ----------
    results : DataFrame ou structure équivalente
        Résultats détaillés du backtest.

    Retour
    ------
    summary : dict
        Dictionnaire contenant par exemple :
        - MAE
        - RMSE
        - erreur moyenne
        - erreur maximale
        - nombre d’observations testées

    Ce qu’il faut faire
    -------------------
    1. Extraire les colonnes utiles depuis les résultats
    2. Calculer les métriques globales
    3. Retourner un résumé simple à afficher dans les notebooks
    """
    predicted = results["prix_prédit"].dropna()

    if "prix_observé" not in results.columns:
        return {
            "n_observations": len(predicted),
            "prix_moyen_predit": predicted.mean(),
            "prix_min_predit": predicted.min(),
            "prix_max_predit": predicted.max(),
            "sigma_moyen": results["sigma"].mean(),
            "p_star_moyen": results["p_star"].mean(),
            "n_nan": results["prix_prédit"].isna().sum(),
        }
    observed = results["prix_observé"].dropna()
    mae = compute_mae(predicted, observed)
    rmse = compute_rmse(predicted, observed)
    errors = predicted.values - observed.values

    return {
        "n_observations": len(predicted),
        "MAE": mae,
        "RMSE": rmse,
        "erreur_moyenne": np.mean(errors),
        "erreur_max": np.max(np.abs(errors)),
        "n_nan": results["prix_prédit"].isna().sum()
    }
    pass


# ============================================================
# 7. ANALYSE DE STABILITE DANS LE TEMPS
# ============================================================

def analyze_stability(results):
    """
    Étudier la stabilité du modèle dans le temps.

    Paramètres
    ----------
    results : DataFrame ou structure équivalente
        Résultats du backtest.

    Retour
    ------
    stability_info : dict ou DataFrame
        Informations permettant d’analyser :
        - l’évolution des erreurs
        - les périodes où le modèle fonctionne mieux ou moins bien
        - la dispersion des résultats

    Ce qu’il faut faire
    -------------------
    1. Observer l’évolution temporelle des erreurs
    2. Identifier les zones de forte ou faible erreur
    3. Préparer des données exploitables pour des graphiques dans les notebooks

    Objectif
    --------
    Ne pas se limiter à une métrique globale,
    mais comprendre comment le modèle se comporte au fil du temps.
    """
    df = results.copy()

    # Evolution de la volatilité calibrée
    sigma_evolution = {
        "sigma_moyen": df["sigma"].mean(),
        "sigma_min": df["sigma"].min(),
        "sigma_max": df["sigma"].max(),
        "sigma_std": df["sigma"].std()
    }

    # Evolution de p_star
    p_star_stats = {
        "p_star_moyen": df["p_star"].mean(),
        "p_star_std": df["p_star"].std()
    }

    # Rolling stats sur le prix prédit (fenêtre 21 jours)
    df["rolling_mean_prix"] = df["prix_prédit"].rolling(21).mean()
    df["rolling_std_prix"] = df["prix_prédit"].rolling(21).std()

    #  Si erreurs, il faut aussi les analyser
    if "erreur_absolue" in df.columns:
        df["rolling_mae"] = df["erreur_absolue"].rolling(21).mean()
        high_error_threshold = df["erreur_absolue"].quantile(0.9)
        df["high_error_period"] = df["erreur_absolue"] > high_error_threshold

    return {
        "sigma_stats": sigma_evolution,
        "p_star_stats": p_star_stats,
        "serie_temporelle": df, # exploitable directement dans les notebooks pour faire des graphiques
    }
    pass


# ============================================================
# 8. FONCTION OPTIONNELLE : PRIX DE REFERENCE
# ============================================================

def build_reference_prices(option_market_data=None, fallback_method=None):
    """
    Construire ou récupérer les prix de référence utilisés pour la comparaison.

    Paramètres
    ----------
    option_market_data : DataFrame or None
        Données réelles de marché sur les options, si disponibles.

    fallback_method : callable or None
        Méthode alternative de référence si les prix d’options observés
        ne sont pas disponibles.

    Retour
    ------
    reference_prices : array-like
        Série de prix servant de référence pour le backtest.

    Remarque
    --------
    Dans un projet étudiant, il est possible que les vrais prix d’options
    ne soient pas facilement disponibles.
    Dans ce cas, il faudra bien expliquer dans le rapport
    ce qui est utilisé comme référence.
    """
    if option_market_data is not None:
        if isinstance(option_market_data, pd.DataFrame):
            if "Close" in option_market_data.columns:
                return option_market_data["Close"].values
            raise ValueError("Colonne 'Close' introuvable dans option_market_data")
        return np.array(option_market_data)

    if fallback_method is not None:
        return fallback_method()

    raise ValueError(
        "Aucune donnée de marché ni méthode de fallback fournie. "
        "Dans un contexte étudiant, tu peux utiliser Black-Scholes comme référence : "
        "fallback_method=lambda: black_scholes_prices(...)"
    )
    pass


# ============================================================
# 9. NOTES D’UTILISATION
# ============================================================

"""
Utilisation typique
-------------------
1. Préparer les données de marché avec market_data.py
2. Définir une option avec option.py
3. Calibrer les paramètres avec calibration.py
4. Calculer des prix avec crr.py
5. Utiliser backtester.py pour évaluer le modèle sur plusieurs dates

Exemple logique
---------------
- on dispose d’une série de dates
- à chaque date, on calibre le modèle
- on calcule le prix théorique
- on compare à une référence
- on mesure les erreurs globales

Remarque importante
-------------------
Ce module ne doit pas refaire toute la logique de pricing.
Il doit orchestrer l’évaluation du modèle, pas remplacer crr.py.

Séparation des rôles
--------------------
- market_data.py : récupération et préparation des données
- calibration.py : estimation des paramètres
- option.py : définition de l’option
- crr.py : pricing théorique
- backtester.py : évaluation empirique du modèle
"""