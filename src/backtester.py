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

# import numpy as np
# import pandas as pd


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
    pass


# ============================================================
# 5. BACKTEST SUR PLUSIEURS DATES
# ============================================================

def run_backtest(data, option, pricing_function, calibration_function, n_steps, risk_free_rate=None):
    """
    Exécuter un backtest du modèle sur plusieurs dates.

    Paramètres
    ----------
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