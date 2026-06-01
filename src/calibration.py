"""
calibration.py

Description
-----------
Ce module est responsable de la calibration des paramètres du modèle
de Cox-Ross-Rubinstein (CRR).

À partir des données de marché, il permet de :
- gérer le taux sans risque
- estimer la volatilité historique
- calculer les paramètres du modèle binomial :
    - u : facteur de hausse
    - d : facteur de baisse
    - p_star : probabilité risque-neutre

Ce module est utilisé par :
- crr.py pour construire l’arbre binomial
- les notebooks d’analyse pour étudier la sensibilité du modèle
"""


# ============================================================
# IMPORTS
# ============================================================

import numpy as np


# ============================================================
# 1. ESTIMATION DE LA VOLATILITE
# ============================================================

def estimate_volatility(log_returns, annualization_factor=252):
    """
    Estimer la volatilité historique annualisée à partir des rendements log.

    Paramètres
    ----------
    log_returns : Series ou array-like
        Série des rendements logarithmiques.

    annualization_factor : int, optional
        Nombre de périodes de marché par an.
        Pour des données journalières, on prend généralement 252.

    Retour
    ------
    sigma : float
        Estimation de la volatilité annualisée.

    Ce qu’il faut faire
    -------------------
    1. Calculer l’écart-type des rendements log
    2. Annualiser cet écart-type
    3. Retourner la volatilité historique

    Formule
    -------
    sigma = std(log_returns) * sqrt(annualization_factor)

    Remarque
    --------
    Cette volatilité sera utilisée dans le calcul de u et d.
    """
    return np.std(log_returns, ddof=1) * np.sqrt(annualization_factor) # ddof=1 pour l'estimation de l'écart-type à partir d'un échantillon 

    



# ============================================================
# 2. GESTION DU TAUX SANS RISQUE
# ============================================================

DEFAULT_RISK_FREE_RATE = 0.01  # 1% par an, à ajuster selon les conditions de marché
def get_risk_free_rate(rate=None):
    """
    Fournir le taux sans risque utilisé dans le modèle.

    Paramètres
    ----------
    rate : float or None
        Taux sans risque fixé manuellement.

    Retour
    ------
    r : float
        Taux sans risque.

    Ce qu’il faut faire
    -------------------
    1. Si un taux est fourni, le retourner directement
    2. Sinon, utiliser une valeur fixée par défaut pour le projet
    3. Vérifier que le taux est cohérent

    Remarque
    --------
    Dans une première version du projet, ce taux peut être donné
    manuellement pour simplifier la calibration.
    """
    r = rate if rate is not None else DEFAULT_RISK_FREE_RATE
    if r < 0 :
        raise ValueError("Le taux sans risque ne peut pas être négatif.")
    return r
    


# ============================================================
# 3. PAS DE TEMPS
# ============================================================

def compute_time_step(maturity, n_steps):
    """
    Calculer le pas de temps du modèle binomial.

    Paramètres
    ----------
    maturity : float
        Maturité de l’option en années.

    n_steps : int
        Nombre d’étapes dans l’arbre binomial.

    Retour
    ------
    dt : float
        Taille d’un pas de temps.

    Formule
    -------
    dt = maturity / n_steps

    Utilité
    -------
    Ce paramètre intervient dans le calcul de u, d et p_star.
    """
    if maturity <= 0:
        raise ValueError(f"La maturité doit être positive. Valeur fournie : {maturity}")
    if n_steps <= 0:
        raise ValueError(f"Nombre d'étape invalide : {n_steps}.")
    return maturity / n_steps
    


# ============================================================
# 4. CALCUL DE u ET d
# ============================================================

def compute_ud(sigma, dt):
    """
    Calculer les facteurs de hausse et de baisse du modèle CRR.

    Paramètres
    ----------
    sigma : float
        Volatilité annualisée.

    dt : float
        Pas de temps.

    Retour
    ------
    u : float
        Facteur de hausse.

    d : float
        Facteur de baisse.

    Formules
    --------
    u = exp(sigma * sqrt(dt))
    d = exp(-sigma * sqrt(dt))

    Remarque
    --------
    Dans le modèle CRR, on a aussi :
    d = 1 / u
    """
    if sigma <= 0:
        raise ValueError(f"Volatilité invalide : {sigma}")
    if dt <= 0:
        raise ValueError(f"Pas de temps invalide : {dt}")
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    return u, d



# ============================================================
# 5. CALCUL DE LA PROBABILITE RISQUE-NEUTRE
# ============================================================

def compute_risk_neutral_probability(r, dt, u, d):
    """
    Calculer la probabilité risque-neutre p_star du modèle CRR.

    Paramètres
    ----------
    r : float
        Taux sans risque.

    dt : float
        Pas de temps.

    u : float
        Facteur de hausse.

    d : float
        Facteur de baisse.

    Retour
    ------
    p_star : float
        Probabilité risque-neutre.

    Formule
    -------
    p_star = (exp(r * dt) - d) / (u - d)

    Ce qu’il faut vérifier
    ----------------------
    1. Le dénominateur ne doit pas être nul
    2. La probabilité obtenue doit être entre 0 et 1

    Remarque
    --------
    Cette probabilité est utilisée dans la rétropropagation
    pour calculer le prix de l’option.
    """
    if u == d:
        raise ValueError("u et d sont égaux, dénominateur nul")
    p_star = (np.exp(r * dt) - d) / (u - d)
    if not (0 <= p_star <= 1):
        raise ValueError(f"Probabilité risque-neutre invalide : {p_star}. Doit être entre 0 et 1.")
    return p_star



# ============================================================
# 6. PIPELINE COMPLET DE CALIBRATION
# ============================================================

def calibrate_crr_parameters(log_returns, maturity, n_steps, risk_free_rate=None):
    """
    Calibrer tous les paramètres nécessaires au modèle CRR.

    Paramètres
    ----------
    log_returns : Series ou array-like
        Rendements logarithmiques de l’actif.

    maturity : float
        Maturité de l’option en années.

    n_steps : int
        Nombre d’étapes de l’arbre binomial.

    risk_free_rate : float or None
        Taux sans risque fixé manuellement.
        Si None, utiliser une valeur par défaut.

    Retour
    ------
    params : dict
        Dictionnaire contenant :
        - sigma
        - r
        - dt
        - u
        - d
        - p_star

    Ce qu’il faut faire
    -------------------
    1. Estimer la volatilité
    2. Définir le taux sans risque
    3. Calculer le pas de temps
    4. Calculer u et d
    5. Calculer p_star
    6. Retourner tous les paramètres dans une structure simple

    Pourquoi retourner un dictionnaire
    ----------------------------------
    Cela permet une utilisation simple dans crr.py et dans les notebooks :
    - params["sigma"]
    - params["u"]
    - params["d"]
    - params["p_star"]
    """
    sigma = estimate_volatility(log_returns)
    r = get_risk_free_rate(risk_free_rate)
    dt = compute_time_step(maturity, n_steps)
    u, d = compute_ud(sigma, dt)
    p_star = compute_risk_neutral_probability(r, dt, u, d)
    return {
        "sigma": sigma,
        "r": r,
        "dt": dt,
        "u": u,
        "d": d,
        "p_star": p_star
    }



# ============================================================
# 7. FONCTION OPTIONNELLE : INTERVALLE DE CONFIANCE
# ============================================================

def volatility_confidence_interval(log_returns, confidence_level=0.95):
    """
    Estimer un intervalle de confiance pour la volatilité.

    Paramètres
    ----------
    log_returns : Series ou array-like
        Rendements logarithmiques.
    confidence_level : float
        Niveau de confiance souhaité.

    Retour
    ------
    interval : tuple
        Borne inférieure et borne supérieure de l’intervalle.

    Remarque
    --------
    Cette fonction est optionnelle.
    Elle peut être utile pour enrichir l’analyse dans le rapport,
    mais elle n’est pas indispensable à la première version du projet.
    """
    from scipy import stats
    n = len(log_returns)
    if n < 2:
        raise ValueError("Il faut au moins deux rendements pour estimer la volatilité.")
    alpha = 1 - confidence_level
    chi2_low = stats.chi2.ppf(alpha / 2, df=n-1)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, df=n-1)
    # intervalle sur la variance, puis on repasse sur la volatilité annualisée
    var = np.var(log_returns, ddof=1)
    lower_bound = np.sqrt((n - 1) * var / chi2_high) * np.sqrt(252)
    upper_bound = np.sqrt((n - 1) * var / chi2_low) * np.sqrt(252)    
    return lower_bound, upper_bound



# ============================================================
# 8. NOTES D’UTILISATION
# ============================================================

"""
Utilisation typique
-------------------
1. Récupérer les rendements log depuis market_data.py
2. Choisir une maturité et un nombre d’étapes
3. Calibrer les paramètres du modèle CRR
4. Passer ces paramètres à crr.py

Exemple logique
---------------
log_returns -> estimate_volatility -> compute_ud -> compute_risk_neutral_probability

Remarque importante
-------------------
Ce module ne fait pas le pricing de l’option.
Il prépare uniquement les paramètres nécessaires au modèle.

Séparation des rôles
--------------------
- market_data.py : récupère et prépare les données
- calibration.py : estime les paramètres
- crr.py : calcule le prix de l’option
"""