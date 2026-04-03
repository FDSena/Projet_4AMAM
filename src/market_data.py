"""
market_data.py

Description
-----------
Ce module gère les données de marché nécessaires au projet.

Son rôle est de :
- télécharger les données financières depuis Yahoo Finance
- nettoyer les données brutes
- extraire les prix utiles
- calculer les rendements logarithmiques
- fournir des données prêtes à être utilisées dans :
    - calibration.py
    - crr.py
    - backtester.py
    - les notebooks d’analyse (.ipynb)

Choix de conception
-------------------
Ce module est construit avec des fonctions simples plutôt qu’avec une classe,
car cela est plus pratique pour :
- les tests rapides
- l’exploration dans les notebooks
- la réutilisation dans tout le projet
"""


# ============================================================
# IMPORTS
# ============================================================

# import pandas as pd
# import numpy as np
# import yfinance as yf


# ============================================================
# 1. TELECHARGEMENT DES DONNEES
# ============================================================

def download_data(ticker, start_date, end_date, interval="1d"):
    """
    Télécharger les données de marché d’un actif depuis Yahoo Finance.

    Paramètres
    ----------
    ticker : str
        Symbole de l’actif financier.
        Exemples :
        - '^FCHI' pour le CAC 40
        - 'AAPL' pour Apple
        - 'BZ=F' pour le Brent

    start_date : str
        Date de début de récupération des données.

    end_date : str
        Date de fin de récupération des données.

    interval : str, optional
        Fréquence des données.
        Par défaut : '1d' pour des données journalières.

    Retour
    ------
    data : DataFrame
        Tableau contenant les données de marché brutes :
        - Open
        - High
        - Low
        - Close
        - Volume
        avec un index de type date.

    Ce qu’il faut faire
    -------------------
    1. Interroger Yahoo Finance via yfinance
    2. Vérifier que les données téléchargées ne sont pas vides
    3. Mettre l’index au bon format temporel
    4. Trier les données par ordre chronologique
    5. Retourner le DataFrame brut
    """
    pass


# ============================================================
# 2. NETTOYAGE DES DONNEES
# ============================================================

def clean_data(data):
    """
    Nettoyer les données de marché téléchargées.

    Paramètres
    ----------
    data : DataFrame
        Données brutes téléchargées.

    Retour
    ------
    clean_df : DataFrame
        Données nettoyées et prêtes à être exploitées.

    Ce qu’il faut faire
    -------------------
    1. Vérifier la présence de valeurs manquantes (NaN)
    2. Supprimer ou corriger les lignes problématiques
    3. Vérifier qu’il n’y a pas de doublons de dates
    4. Vérifier l’ordre chronologique des observations
    5. S’assurer que les colonnes utiles sont bien présentes

    Remarque
    --------
    Le nettoyage doit rester simple et robuste.
    L’objectif est d’obtenir une série exploitable pour les calculs de rendement
    et de volatilité.
    """
    pass


# ============================================================
# 3. EXTRACTION DES PRIX DE CLOTURE
# ============================================================

def get_close_prices(data):
    """
    Extraire la série des prix de clôture.

    Paramètres
    ----------
    data : DataFrame
        Données de marché nettoyées.

    Retour
    ------
    close_prices : Series
        Série temporelle des prix de clôture.

    Pourquoi cette fonction
    -----------------------
    Les prix de clôture sont la base la plus naturelle pour :
    - calculer les rendements
    - estimer la volatilité
    - construire des analyses temporelles dans les notebooks
    """
    pass


# ============================================================
# 4. CALCUL DES RENDEMENTS LOGARITHMIQUES
# ============================================================

def compute_log_returns(prices):
    """
    Calculer les rendements logarithmiques à partir d’une série de prix.

    Paramètres
    ----------
    prices : Series
        Série des prix de clôture.

    Retour
    ------
    log_returns : Series
        Série des rendements logarithmiques.

    Formule
    -------
    r_t = log(S_t / S_{t-1})

    Ce qu’il faut faire
    -------------------
    1. Calculer le rapport entre deux prix consécutifs
    2. Appliquer le logarithme
    3. Supprimer la première valeur manquante générée par le décalage
    4. Retourner une série propre

    Utilité
    -------
    Cette série sera utilisée dans calibration.py pour estimer la volatilité.
    """
    pass


# ============================================================
# 5. PIPELINE PRINCIPAL
# ============================================================

def get_market_data(ticker, start_date, end_date, interval="1d"):
    """
    Exécuter le pipeline complet de préparation des données de marché.

    Paramètres
    ----------
    ticker : str
        Symbole de l’actif.
    start_date : str
        Date de début.
    end_date : str
        Date de fin.
    interval : str, optional
        Fréquence des données.

    Retour
    ------
    result : dict
        Structure contenant par exemple :
        - les données brutes
        - les données nettoyées
        - les prix de clôture
        - les rendements log

    Ce qu’il faut faire
    -------------------
    1. Télécharger les données
    2. Nettoyer les données
    3. Extraire les prix de clôture
    4. Calculer les rendements log
    5. Retourner les résultats dans un format simple à réutiliser

    Pourquoi retourner un dictionnaire
    ----------------------------------
    Cela permet aux notebooks d’accéder facilement à chaque étape :
    - result["raw_data"]
    - result["clean_data"]
    - result["close_prices"]
    - result["log_returns"]
    """
    pass


# ============================================================
# 6. FONCTION OPTIONNELLE POUR PLUSIEURS ACTIFS
# ============================================================

def get_multiple_close_prices(tickers, start_date, end_date, interval="1d"):
    """
    Récupérer les prix de clôture de plusieurs actifs et aligner les dates.

    Paramètres
    ----------
    tickers : list[str]
        Liste des symboles des actifs.
    start_date : str
        Date de début.
    end_date : str
        Date de fin.
    interval : str, optional
        Fréquence des données.

    Retour
    ------
    prices_df : DataFrame
        Tableau dont :
        - chaque colonne correspond à un actif
        - chaque ligne correspond à une date alignée

    Ce qu’il faut faire
    -------------------
    1. Télécharger les données pour chaque ticker
    2. Nettoyer chaque série
    3. Extraire les prix de clôture
    4. Aligner toutes les séries sur les mêmes dates
    5. Construire un DataFrame final

    Utilité
    -------
    Cette fonction est utile surtout pour l’extension portefeuille.
    """
    pass


# ============================================================
# 7. NOTES D’UTILISATION
# ============================================================

"""
Utilisation prévue dans les notebooks
-------------------------------------
Dans les notebooks d’analyse, ce module doit permettre par exemple de :
- visualiser l’évolution des prix
- tracer les rendements
- étudier la volatilité historique
- comparer plusieurs actifs

Exemple logique d’utilisation
-----------------------------
1. Télécharger les données
2. Nettoyer les données
3. Extraire les prix
4. Calculer les rendements
5. Faire les graphiques et analyses dans un fichier .ipynb

Remarque importante
-------------------
On sépare volontairement :
- la logique de récupération / préparation dans market_data.py
- l’analyse et la visualisation dans les notebooks

Cela rend le projet plus propre, plus lisible et plus facile à partager.
"""