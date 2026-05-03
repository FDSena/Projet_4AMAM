"""
portfolio_math.py
=================
Briques mathématiques pour l'optimisation mean-variance.

Fondements théoriques
---------------------
On travaille dans le cadre de Markowitz (1952).

Soit un portefeuille de n actifs avec :
  - µ ∈ ℝⁿ   : vecteur des espérances de rendement
  - Σ ∈ ℝⁿˣⁿ : matrice de covariance (symétrique définie positive)
  - w ∈ ℝⁿ   : vecteur des poids  (contrainte : 1ᵀw = 1)

Rendement du portefeuille :
    R_p = wᵀ r  →  E[R_p] = wᵀµ,  Var[R_p] = wᵀΣw

Objectif mean-variance (Lagrangien dual) :
    min_w  λ · wᵀΣw  −  wᵀµ      s.t.  1ᵀw = 1,  w ≥ 0 (optionnel)

Gradient de la fonction de coût (utilisé par sgd_optimizer.py) :
    ∇_w J = 2λΣw − µ         (avant annualisation)

Annualisation (rendements journaliers → annuels) :
    µ_ann  = µ  · T
    Σ_ann  = Σ  · T      avec T = 252 jours ouvrés
    J_ann  = λ · T · wᵀΣw  −  T · wᵀµ
    ∇J_ann = T · (2λΣw − µ)

Estimation statistique
----------------------
On estime µ et Σ par leurs estimateurs de maximum de vraisemblance :
    µ̂ = (1/T) Σ_t r_t                  (moyenne empirique)
    Σ̂ = (1/(T-1)) Σ_t (r_t-µ̂)(r_t-µ̂)ᵀ  (covariance corrigée de Bessel)

Pour la robustesse, on ajoute une régularisation de Ledoit-Wolf optionnelle.
"""

import numpy as np
import pandas as pd


# ============================================================
# 1. ESTIMATION DES ESPÉRANCES DE RENDEMENT
# ============================================================

def estimate_expected_returns(returns_matrix: np.ndarray | pd.DataFrame) -> np.ndarray:
    """
    Estimer µ̂ = E[r] par la moyenne empirique.

    Estimateur : µ̂_i = (1/T) Σ_t r_{t,i}   (maximum de vraisemblance gaussien)

    Paramètres
    ----------
    returns_matrix : (T, n) array ou DataFrame
        T observations, n actifs.

    Retour
    ------
    mu : (n,) ndarray
        Vecteur des rendements moyens.
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    if R.shape[0] < 2:
        raise ValueError("Il faut au moins 2 observations pour estimer µ.")
    return R.mean(axis=0)


# ============================================================
# 2. ESTIMATION DE LA MATRICE DE COVARIANCE
# ============================================================

def estimate_covariance_matrix(
    returns_matrix: np.ndarray | pd.DataFrame,
    shrinkage: float = 0.0
) -> np.ndarray:
    """
    Estimer Σ̂ par la covariance empirique corrigée (Bessel).

    Estimateur sans biais : Σ̂ = (1/(T-1)) Σ_t (r_t - µ̂)(r_t - µ̂)ᵀ

    Régularisation optionnelle (Ledoit-Wolf linéaire) :
        Σ̂_reg = (1-α) Σ̂ + α · (trace(Σ̂)/n) · I
    avec α = shrinkage ∈ [0, 1].
    Cela garantit que Σ̂_reg est définie positive même avec peu d'observations.

    Paramètres
    ----------
    returns_matrix : (T, n)
    shrinkage : float ∈ [0, 1]
        0 = pas de régularisation, 1 = matrice diagonale (variances).

    Retour
    ------
    Sigma : (n, n) ndarray, symétrique semi-définie positive.
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    T, n = R.shape
    if T < 2:
        raise ValueError("Il faut au moins 2 observations pour estimer Σ.")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage doit être dans [0, 1].")

    Sigma = np.cov(R, rowvar=False)          # (n, n), diviseur T-1

    if shrinkage > 0.0:
        mu_diag = np.trace(Sigma) / n        # cible : variance moyenne × I
        Sigma = (1.0 - shrinkage) * Sigma + shrinkage * mu_diag * np.eye(n)

    return Sigma


# ============================================================
# 3. VÉRIFICATION DES POIDS
# ============================================================

def check_weights(weights: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Vérifier qu'un vecteur de poids est numériquement valide.

    Conditions vérifiées :
    1. Non vide
    2. Numérique (pas de NaN, pas d'Inf)
    3. Somme ≈ 1  (portefeuille entièrement investi)

    Paramètres
    ----------
    weights : array-like
    tol : float — tolérance sur |sum(w) - 1|

    Retour
    ------
    bool
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return False
    if not np.isfinite(w).all():
        return False
    return bool(np.isclose(w.sum(), 1.0, atol=tol))


# ============================================================
# 4. NORMALISATION DES POIDS
# ============================================================

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normaliser w ← w / (1ᵀw)  de sorte que 1ᵀw = 1.

    Si 1ᵀw ≈ 0 (vecteur nul), retourne des poids égaux par défaut.
    """
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if np.isclose(s, 0.0):
        return np.ones(len(w)) / len(w)
    return w / s


# ============================================================
# 5. PROJECTION SUR LE SIMPLEXE (long-only)
# ============================================================

def project_to_simplex(weights: np.ndarray) -> np.ndarray:
    """
    Projeter w sur le simplexe unitaire Δ = {w ≥ 0, 1ᵀw = 1}.

    Algorithme exact O(n log n) — Duchi et al. (2008).
    Garantit :  argmin_{v ∈ Δ}  ||v - w||²

    Paramètres
    ----------
    weights : (n,) ndarray

    Retour
    ------
    v : (n,) ndarray  avec v ≥ 0 et 1ᵀv = 1.
    """
    w = np.asarray(weights, dtype=float).copy()
    n = len(w)
    u = np.sort(w)[::-1]               # tri décroissant
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(w - theta, 0.0)


def project_to_nonnegative_weights(weights: np.ndarray) -> np.ndarray:
    """
    Clip négatifs à 0 puis normalise (projection approchée, rapide).
    Utiliser project_to_simplex pour la projection exacte L².
    """
    w = np.asarray(weights, dtype=float).copy()
    w = np.maximum(w, 0.0)
    return normalize_weights(w)


# ============================================================
# 6. RENDEMENT ESPÉRÉ DU PORTEFEUILLE
# ============================================================

def portfolio_expected_return(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    risk_free_rate: float | None = None
) -> float:
    """
    Calculer E[R_p] = wᵀµ.

    Avec actif sans risque (dernier poids = w_rf) :
        E[R_p] = w_risqué ᵀ µ_risqué  +  w_rf · r_f

    Paramètres
    ----------
    weights : (n,) ou (n+1,) si actif sans risque
    expected_returns : (n,)
    risk_free_rate : float or None

    Retour
    ------
    mu_p : float
    """
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)

    if risk_free_rate is None:
        return float(w @ mu)

    # dernier poids = allocation à l'actif sans risque
    return float(w[:-1] @ mu + w[-1] * risk_free_rate)


# ============================================================
# 7. VARIANCE DU PORTEFEUILLE
# ============================================================

def portfolio_variance(
    weights: np.ndarray,
    covariance_matrix: np.ndarray
) -> float:
    """
    Calculer Var[R_p] = wᵀΣw.

    Propriétés garanties :
    - résultat ≥ 0 (Σ semi-définie positive)
    - symétrie exploitée par le produit matriciel

    Paramètres
    ----------
    weights : (n,)
    covariance_matrix : (n, n) symétrique semi-définie positive

    Retour
    ------
    sigma2_p : float ≥ 0
    """
    w = np.asarray(weights, dtype=float)
    Sigma = np.asarray(covariance_matrix, dtype=float)
    n = len(w)
    if Sigma.shape != (n, n):
        raise ValueError(f"Sigma doit être ({n},{n}), reçu {Sigma.shape}.")
    return float(w @ Sigma @ w)


# ============================================================
# 8. VOLATILITÉ DU PORTEFEUILLE
# ============================================================

def portfolio_volatility(
    weights: np.ndarray,
    covariance_matrix: np.ndarray
) -> float:
    """
    Calculer σ_p = √(wᵀΣw).

    Le clip à 0 évite les erreurs numériques si wᵀΣw < 0 par arrondi.
    """
    return float(np.sqrt(max(portfolio_variance(weights, covariance_matrix), 0.0)))


# ============================================================
# 9. RATIO DE SHARPE (EX-ANTE)
# ============================================================

def portfolio_sharpe(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Ratio de Sharpe ex-ante annualisé :

        SR = (µ_p_ann - r_f) / σ_p_ann

    avec µ_p_ann = T · wᵀµ  et  σ_p_ann = √T · σ_p

    Paramètres
    ----------
    weights : (n,)
    expected_returns : (n,)
    covariance_matrix : (n, n)
    risk_free_rate : float — taux sans risque annuel
    annualization_factor : int — T = 252

    Retour
    ------
    sharpe : float  (0.0 si σ_p ≈ 0)
    """
    T = annualization_factor
    mu_p = portfolio_expected_return(weights, expected_returns) * T
    sigma_p = portfolio_volatility(weights, covariance_matrix) * np.sqrt(T)
    if np.isclose(sigma_p, 0.0):
        return 0.0
    return (mu_p - risk_free_rate) / sigma_p


# ============================================================
# 10. FONCTION DE COÛT MEAN-VARIANCE (ANNUALISÉE)
# ============================================================

def mean_variance_cost(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    lambda_risk: float = 1.0,
    annualization_factor: int = 252
) -> float:
    """
    Fonction de coût mean-variance annualisée :

        J(w) = λ · T · wᵀΣw  −  T · wᵀµ

    où T = annualization_factor.

    Minimiser J(w) revient à maximiser l'utilité espérée d'un agent
    avec aversion au risque λ (approximation quadratique de U).

    Paramètres
    ----------
    weights : (n,)
    expected_returns : (n,)
    covariance_matrix : (n, n)
    lambda_risk : float > 0 — coefficient d'aversion au risque
    annualization_factor : int

    Retour
    ------
    cost : float
    """
    T = annualization_factor
    var_p = portfolio_variance(weights, covariance_matrix)
    mu_p  = portfolio_expected_return(weights, expected_returns)
    return lambda_risk * T * var_p - T * mu_p


# ============================================================
# 11. GRADIENT ANALYTIQUE DE J(w)
# ============================================================

def mean_variance_gradient(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    lambda_risk: float = 1.0,
    annualization_factor: int = 252
) -> np.ndarray:
    """
    Gradient analytique exact de mean_variance_cost :

        ∇_w J = T · (2λ Σw − µ)

    Dérivation :
        ∂/∂w [λT · wᵀΣw] = 2λT Σw   (Σ symétrique)
        ∂/∂w [−T · wᵀµ]  = −T µ

    Ce gradient est utilisé directement par run_sgd pour éviter
    le coût O(n²) de l'approximation numérique.

    Paramètres
    ----------
    weights : (n,)
    expected_returns : (n,)
    covariance_matrix : (n, n)
    lambda_risk : float
    annualization_factor : int

    Retour
    ------
    grad : (n,) ndarray
    """
    w   = np.asarray(weights, dtype=float)
    mu  = np.asarray(expected_returns, dtype=float)
    Sig = np.asarray(covariance_matrix, dtype=float)
    T   = annualization_factor
    return T * (2.0 * lambda_risk * Sig @ w - mu)


# ============================================================
# 12. FONCTION DE COÛT AVEC RENDEMENT CIBLE
# ============================================================

def target_return_cost(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    target_return: float,
    alpha: float = 10.0
) -> float:
    """
    Minimiser le risque sous contrainte souple de rendement cible.

    J(w) = wᵀΣw  +  α · (wᵀµ − µ_cible)²

    Le terme de pénalité quadratique α·(E[R_p] - µ_cible)²
    pousse le portefeuille vers le rendement souhaité
    sans résoudre un problème contraint explicitement.

    Paramètres
    ----------
    weights : (n,)
    expected_returns : (n,)
    covariance_matrix : (n, n)
    target_return : float
    alpha : float — coefficient de pénalité (plus α grand → contrainte plus stricte)

    Retour
    ------
    cost : float
    """
    var_p = portfolio_variance(weights, covariance_matrix)
    mu_p  = portfolio_expected_return(weights, expected_returns)
    return var_p + alpha * (mu_p - target_return) ** 2


# ============================================================
# 13. AJOUT D'UN ACTIF SANS RISQUE
# ============================================================

def add_risk_free_asset(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Étendre µ et Σ pour inclure un actif sans risque.

    Propriétés de l'actif sans risque :
        Var[r_f] = 0          → ligne et colonne nulles dans Σ
        Cov[r_f, r_i] = 0    → pas de corrélation avec les actifs risqués

    Retour
    ------
    mu_ext  : (n+1,)   — µ avec r_f en dernière position
    Sig_ext : (n+1, n+1) — Σ étendue (dernière ligne/colonne = 0)
    """
    mu  = np.asarray(expected_returns, dtype=float)
    Sig = np.asarray(covariance_matrix, dtype=float)
    n = len(mu)
    mu_ext  = np.append(mu, risk_free_rate)
    Sig_ext = np.zeros((n + 1, n + 1))
    Sig_ext[:n, :n] = Sig
    return mu_ext, Sig_ext


# ============================================================
# 14. SÉPARATION DES POIDS (risqués / sans risque)
# ============================================================

def split_weights(weights: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Séparer le vecteur de poids en actifs risqués + actif sans risque.

    Convention : le dernier poids est celui de l'actif sans risque.

    Retour
    ------
    (w_risky, w_rf) : ((n,), float)
    """
    w = np.asarray(weights, dtype=float)
    if len(w) < 2:
        raise ValueError("Il faut au moins 2 poids pour effectuer la séparation.")
    return w[:-1], float(w[-1])


# ============================================================
# 15. CONTRAINTES DE BASE
# ============================================================

def enforce_portfolio_constraints(
    weights: np.ndarray,
    nonnegative: bool = True,
    exact_simplex: bool = False
) -> np.ndarray:
    """
    Appliquer les contraintes du portefeuille.

    Deux modes :
    - exact_simplex=True  : projection exacte sur Δ (optimal au sens L²)
    - exact_simplex=False : clip + normalisation (plus rapide)

    Paramètres
    ----------
    weights : (n,)
    nonnegative : bool — interdit vente à découvert
    exact_simplex : bool — utilise l'algorithme de Duchi et al.

    Retour
    ------
    w_constrained : (n,)
    """
    w = np.asarray(weights, dtype=float)
    if nonnegative:
        if exact_simplex:
            return project_to_simplex(w)
        return project_to_nonnegative_weights(w)
    return normalize_weights(w)