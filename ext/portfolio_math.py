"""
portfolio_math.py

Description
-----------
Ce module regroupe les outils mathématiques nécessaires
à l’optimisation d’un portefeuille multi-actifs.

Son rôle est de :
- définir les poids du portefeuille
- calculer le rendement espéré
- calculer le risque (variance / volatilité)
- intégrer éventuellement un actif sans risque
- construire une fonction de coût de type mean-variance
- gérer les contraintes sur les poids

Ce module appartient à la partie extension du projet.
Il sera utilisé par :
- sgd_optimizer.py
- portfolio_backtest.py
- les notebooks d’analyse
"""


# ============================================================
# IMPORTS
# ============================================================

# import numpy as np


# ============================================================
# 1. RENDEMENTS MOYENS DES ACTIFS
# ============================================================

import pandas as pd
import numpy as np

def estimate_expected_returns(returns_matrix):
    """
    Estimer le rendement moyen de chaque actif.

    Paramètres
    ----------
    returns_matrix : array-like or DataFrame
        Matrice des rendements historiques.
        Chaque colonne représente un actif,
        chaque ligne représente une date.

    Retour
    ------
    mu : array-like
        Vecteur des rendements moyens estimés pour chaque actif.

    Ce qu’il faut faire
    -------------------
    1. Calculer la moyenne des rendements pour chaque actif
    2. Retourner un vecteur de dimension égale au nombre d’actifs

    Utilité
    -------
    Ce vecteur servira au calcul du rendement espéré du portefeuille.
    """
    if not isinstance(returns_matrix, pd.DataFrame):
        returns_matrix = pd.DataFrame(returns_matrix)
    
    mu = returns_matrix.mean(axis=0)

    return mu.values


# ============================================================
# 2. MATRICE DE COVARIANCE
# ============================================================

def estimate_covariance_matrix(returns_matrix):
    """
    Estimer la matrice de covariance des rendements.

    Paramètres
    ----------
    returns_matrix : array-like or DataFrame
        Matrice des rendements des actifs.

    Retour
    ------
    covariance_matrix : array-like
        Matrice de covariance des rendements.

    Ce qu’il faut faire
    -------------------
    1. Utiliser les rendements historiques
    2. Calculer la covariance entre les actifs
    3. Retourner une matrice carrée

    Utilité
    -------
    Cette matrice sert à quantifier le risque du portefeuille.
    """
    if not isinstance(returns_matrix, pd.DataFrame):
        returns_matrix = pd.DataFrame(returns_matrix)

    covariance_matrix = returns_matrix.cov()

    return covariance_matrix.values
    
# ============================================================
# 3. VERIFICATION DES POIDS
# ============================================================

def check_weights(weights):
    """
    Vérifier qu’un vecteur de poids est cohérent.

    Paramètres
    ----------
    weights : array-like
        Vecteur des poids du portefeuille.

    Retour
    ------
    valid : bool
        Indique si les poids sont acceptables.

    Ce qu’il faut vérifier
    ----------------------
    1. Les poids doivent être numériques
    2. Le vecteur ne doit pas être vide
    3. Les poids ne doivent pas contenir de NaN
    4. La somme des poids doit pouvoir être contrôlée

    Remarque
    --------
    Cette fonction sert de garde-fou avant tout calcul.
    """
    weights = np.array(weights)

    if weights.size == 0:
        return False

    if not np.issubdtype(weights.dtype, np.number):
        return False

    if np.isnan(weights).any() or np.isinf(weights).any():
        return False

    total_weight = np.sum(weights)

    if not (np.isclose(total_weight, 1.0) or np.isclose(total_weight, 0.0)):
        return False

    return True
# ============================================================
# 4. NORMALISATION DES POIDS
# ============================================================

def normalize_weights(weights):
    """
    Normaliser les poids pour que leur somme soit égale à 1.

    Paramètres
    ----------
    weights : array-like
        Vecteur de poids.

    Retour
    ------
    normalized_weights : array-like
        Vecteur de poids renormalisé.

    Formule
    -------
    w_normalized = w / sum(w)

    Utilité
    -------
    Garantit que le portefeuille est entièrement investi.
    """
    weights = np.array(weights, dtype=float)
    total_weight = np.sum(weights)
    if total_weight == 0:
        return weights
    normalized_weights = weights / total_weight
    return normalized_weights

# ============================================================
# 5. PROJECTION DES POIDS POSITIFS
# ============================================================

def project_to_nonnegative_weights(weights):
    """
    Projeter les poids pour éviter les valeurs négatives.

    Paramètres
    ----------
    weights : array-like
        Vecteur de poids.

    Retour
    ------
    projected_weights : array-like
        Vecteur de poids avec contraintes de positivité.

    Ce qu’il faut faire
    -------------------
    1. Remplacer ou corriger les poids négatifs
    2. Renormaliser si nécessaire

    Utilité
    -------
    Permet d’imposer l’absence de vente à découvert.
    """
    weights = np.array(weights, dtype=float)
    weights[weights < 0] = 0.0
    return normalize_weights(weights)

# ============================================================
# 6. RENDEMENT ESPERE DU PORTEFEUILLE
# ============================================================

def portfolio_expected_return(weights, expected_returns, risk_free_rate=None):
    """
    Calculer le rendement espéré du portefeuille.

    Paramètres
    ----------
    weights : array-like
        Vecteur des poids du portefeuille.
    expected_returns : array-like
        Rendements moyens des actifs risqués.
    risk_free_rate : float or None
        Taux sans risque, si un actif sans risque est intégré explicitement.

    Retour
    ------
    mu_p : float
        Rendement espéré du portefeuille.

    Formule
    -------
    mu_p = w^T * mu

    Cas avec actif sans risque
    --------------------------
    Si l’actif sans risque est inclus dans le vecteur de poids,
    sa contribution doit être prise en compte explicitement.

    Utilité
    -------
    C’est l’un des deux termes centraux du modèle mean-variance.
    """
    weights = np.array(weights,dtype=float)
    expected_returns = np.array(expected_returns,dtype=float)
    
    if risk_free_rate is None:
        mu_p = np.dot(weights, expected_returns)
        return mu_p
    
    else:
        risky_weights = weights[:-1]
        w_rf = weights[-1]
        mu_risky = np.dot(risky_weights, expected_returns)
        mu_rf = w_rf * risk_free_rate
        mu_p = mu_risky + mu_rf
        return mu_p

# ============================================================
# 7. VARIANCE DU PORTEFEUILLE
# ============================================================

def portfolio_variance(weights, covariance_matrix):
    """
    Calculer la variance du portefeuille.

    Paramètres
    ----------
    weights : array-like
        Vecteur des poids.
    covariance_matrix : array-like
        Matrice de covariance des rendements.

    Retour
    ------
    sigma_p2 : float
        Variance du portefeuille.

    Formule
    -------
    sigma_p² = w^T * Sigma * w

    Utilité
    -------
    Mesure principale du risque dans l’approche mean-variance.
    """
    weights = np.array(weights, dtype=float)
    covariance_matrix = np.array(covariance_matrix, dtype=float)

    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if covariance_matrix.shape[0] != weights.size:
        raise ValueError("Covariance matrix size must match number of weights.")
    
    # calcul variance
    sigma_p2 = weights.T @ covariance_matrix @ weights

    return sigma_p2

 


# ============================================================
# 8. VOLATILITE DU PORTEFEUILLE
# ============================================================

def portfolio_volatility(weights, covariance_matrix):
    """
    Calculer la volatilité du portefeuille.

    Paramètres
    ----------
    weights : array-like
        Vecteur des poids.
    covariance_matrix : array-like
        Matrice de covariance.

    Retour
    ------
    sigma_p : float
        Volatilité du portefeuille.

    Formule
    -------
    sigma_p = sqrt(w^T * Sigma * w)

    Utilité
    -------
    Donne une mesure du risque plus interprétable que la variance.
    """
    variance = portfolio_variance(weights, covariance_matrix)
    variance = max(variance, 0.0)
    sigma_p = np.sqrt(variance)
    return sigma_p


# ============================================================
# 9. FONCTION DE COUT MEAN-VARIANCE
# ============================================================

def mean_variance_cost(weights, expected_returns, covariance_matrix, lambda_risk=1.0):
    """
    Construire une fonction de coût de type mean-variance.

    Paramètres
    ----------
    weights : array-like
        Vecteur des poids.
    expected_returns : array-like
        Rendements moyens estimés.
    covariance_matrix : array-like
        Matrice de covariance.
    lambda_risk : float
        Paramètre d’aversion au risque.

    Retour
    ------
    cost : float
        Valeur de la fonction de coût.

    Idée générale
    -------------
    La fonction de coût peut pénaliser :
    - le risque du portefeuille
    - et récompenser le rendement espéré

    Forme possible
    --------------
    J(w) = lambda_risk * variance - expected_return

    Remarque
    --------
    La forme exacte doit être choisie de manière cohérente
    avec l’objectif du projet.
    """
    expected_return = portfolio_expected_return(weights, expected_returns)
    variance = portfolio_variance(weights, covariance_matrix)

    cost = lambda_risk * variance - expected_return

    return cost


# ============================================================
# 10. VERSION AVEC RENDEMENT CIBLE
# ============================================================

def target_return_cost(weights, expected_returns, covariance_matrix, target_return, alpha=10):
    """
    Construire une fonction de coût pour minimiser le risque
    sous contrainte d’un rendement cible.

    Paramètres
    ----------
    weights : array-like
        Vecteur des poids.
    expected_returns : array-like
        Rendements moyens estimés.
    covariance_matrix : array-like
        Matrice de covariance.
    target_return : float
        Niveau de rendement visé.

    Retour
    ------
    cost : float
        Valeur de la fonction de coût.

    Idée générale
    -------------
    On cherche un portefeuille peu risqué
    tout en restant proche d’un rendement cible.

    Remarque
    --------
    Cette fonction peut être utile pour comparer
    plusieurs formulations de l’optimisation.
    """
    expected_return = portfolio_expected_return(weights, expected_returns)
    variance = portfolio_variance(weights, covariance_matrix)

    return_penalty = (expected_return - target_return) ** 2

    # coefficient de pénalité
    cost = variance + alpha * return_penalty

    return cost

# ============================================================
# 11. AJOUT EXPLICITE D’UN ACTIF SANS RISQUE
# ============================================================

def add_risk_free_asset(expected_returns, covariance_matrix, risk_free_rate):
    """
    Ajouter explicitement un actif sans risque au portefeuille.

    Paramètres
    ----------
    expected_returns : array-like
        Rendements moyens des actifs risqués.
    covariance_matrix : array-like
        Matrice de covariance des actifs risqués.
    risk_free_rate : float
        Taux sans risque.

    Retour
    ------
    new_expected_returns : array-like
        Vecteur de rendements avec actif sans risque ajouté.
    new_covariance_matrix : array-like
        Matrice de covariance élargie.

    Ce qu’il faut faire
    -------------------
    1. Ajouter le rendement sans risque au vecteur des espérances
    2. Ajouter une ligne et une colonne de covariance nulle
    3. Retourner les nouvelles structures

    Remarque
    --------
    Un actif sans risque a une variance nulle
    et une covariance nulle avec les actifs risqués
    dans cette modélisation simplifiée.
    """
    expected_returns = np.array(expected_returns, dtype=float)
    covariance_matrix = np.array(covariance_matrix, dtype=float)

    n = expected_returns.shape[0]

    new_expected_returns = np.append(expected_returns, risk_free_rate)

    new_covariance_matrix = np.zeros((n + 1, n + 1))
    new_covariance_matrix[:n, :n] = covariance_matrix

    return new_expected_returns, new_covariance_matrix

# ============================================================
# 12. CONTRAINTES DE BASE
# ============================================================

def enforce_portfolio_constraints(weights, nonnegative=True):
    """
    Appliquer les contraintes de base sur les poids du portefeuille.

    Paramètres
    ----------
    weights : array-like
        Vecteur des poids.
    nonnegative : bool
        Si True, interdit les poids négatifs.

    Retour
    ------
    constrained_weights : array-like
        Vecteur de poids corrigé.

    Ce qu’il faut faire
    -------------------
    1. Imposer éventuellement la positivité
    2. Renormaliser pour que la somme fasse 1

    Utilité
    -------
    Cette fonction sera utile après chaque mise à jour dans sgd_optimizer.py
    """
    weights = np.array(weights, dtype=float)

    if nonnegative:
        return project_to_nonnegative_weights(weights)

    return normalize_weights(weights)

# ============================================================
# 13. NOTES D’UTILISATION
# ============================================================

"""
Utilisation typique
-------------------
1. Récupérer les prix de plusieurs actifs avec market_data.py
2. Calculer leurs rendements
3. Estimer :
    - les rendements moyens
    - la matrice de covariance
4. Définir un portefeuille via un vecteur de poids
5. Calculer :
    - le rendement espéré
    - le risque
    - la fonction de coût
6. Optimiser les poids avec sgd_optimizer.py

Exemple logique
---------------
returns_matrix -> expected_returns + covariance_matrix -> portfolio metrics -> cost function

Remarque importante
-------------------
Ce module ne réalise pas l’optimisation lui-même.
Il fournit uniquement les briques mathématiques nécessaires.

Séparation des rôles
--------------------
- market_data.py : fournit les données
- portfolio_math.py : définit les formules mathématiques
- sgd_optimizer.py : optimise les poids
- portfolio_backtest.py : évalue les stratégies obtenues
"""