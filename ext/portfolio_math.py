"""
portfolio_math.py
=================
Briques mathématiques pour l'optimisation mean-variance d'un portefeuille
composé de plusieurs actifs risqués et d'un actif sans risque.

Convention générale
-------------------
Les données de marché contiennent uniquement les n actifs risqués.

- r_t in R^n         : rendements journaliers des actifs risqués
- mu in R^n          : moyenne journalière estimée des actifs risqués
- Sigma in R^{n x n} : covariance journalière estimée des actifs risqués
- w in R^n           : poids investis dans les actifs risqués
- w_rf               : poids investi dans l'actif sans risque

La contrainte budgétaire complète est :

    sum_i w_i + w_rf = 1

Dans ce projet, on n'optimise pas directement w_rf. On optimise uniquement
les poids risqués w, puis on déduit le poids de l'actif sans risque comme
résidu budgétaire :

    w_rf = 1 - sum_i w_i

Pour que l'actif sans risque puisse recevoir une allocation non nulle, on
n'impose PAS sum(w)=1, mais seulement, dans le cas long-only :

    w_i >= 0
    sum_i w_i <= 1

Ainsi, si sum_i w_i < 1, la fraction restante est placée dans l'actif sans
risque. Si sum_i w_i = 1, le portefeuille est entièrement investi en actifs
risqués (w_rf = 0).

Annualisation et cohérence dimensionnelle
------------------------------------------
Les rendements des actifs risqués sont supposés journaliers.
Le taux sans risque risk_free_rate est supposé ANNUEL.

Sous l'hypothèse i.i.d. :

    mu_ann    = T * mu        (T = 252 jours de bourse)
    Sigma_ann = T * Sigma

La fonction de coût mean-variance avec cash mélange délibérément un rendement
risqué annualisé (T * w^T mu) et un rendement sans risque annuel (w_rf * r_f).
Ces deux termes sont homogènes : tous deux expriment un rendement annuel.
Le terme T * w^T mu annualise la moyenne journalière des actifs risqués, et
r_f est directement annuel. Il n'y a donc pas d'incohérence dimensionnelle,
à condition de ne jamais mélanger des mu journaliers non annualisés avec r_f
annuel dans la même expression.

Fonction de coût sans actif sans risque explicite
-------------------------------------------------
Cette version correspond à un portefeuille entièrement investi dans les actifs
risqués, avec sum(w)=1 :

    J(w) = lambda * T * w^T Sigma w - T * w^T mu

Gradient :

    grad J(w) = T * (2 * lambda * Sigma w - mu)

Fonction de coût avec actif sans risque comme résidu (VERSION PRINCIPALE)
--------------------------------------------------------------------------
On définit w_rf = 1 - sum(w), puis :

    J_cash(w) = lambda * T * w^T Sigma w
                - [T * w^T mu + (1 - sum(w)) * r_f]

Tous les termes sont exprimés en rendement annuel :
  - T * w^T mu     : rendement annualisé de la poche risquée (mu journalier * T)
  - (1-sum(w))*r_f : rendement annuel de la poche sans risque (r_f déjà annuel)
  - T * w^T Sigma w: variance annualisée (Sigma journalière * T)

Gradient analytique (obtenu par différentiation directe) :

    d/dw_i [-(1 - sum(w)) * r_f] = d/dw_i [r_f * sum(w)] = r_f

    grad J_cash(w) = T * (2 * lambda * Sigma w - mu) + r_f * 1_n

Cette formulation permet au SGD d'arbitrer entre actifs risqués et actif sans
risque, à condition de projeter w sur {w >= 0, sum(w) <= 1} après chaque pas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# 1. VALIDATION DES ENTRÉES
# ============================================================

def _as_2d_returns(returns_matrix: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Convertir une matrice de rendements en ndarray 2D de type float."""
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    if R.ndim != 2:
        raise ValueError("returns_matrix doit être un tableau 2D de forme (T, n).")
    if not np.isfinite(R).all():
        raise ValueError("returns_matrix contient NaN ou Inf.")
    return R


def _validate_weights_and_mu(
    weights: np.ndarray, expected_returns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Valider que w et mu sont deux vecteurs 1D de même dimension."""
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)

    if w.ndim != 1 or mu.ndim != 1:
        raise ValueError("weights et expected_returns doivent être des vecteurs 1D.")
    if w.shape != mu.shape:
        raise ValueError(
            f"weights {w.shape} et expected_returns {mu.shape} doivent avoir la même forme."
        )
    if not np.isfinite(w).all() or not np.isfinite(mu).all():
        raise ValueError("weights ou expected_returns contient NaN ou Inf.")
    return w, mu


def _validate_covariance(covariance_matrix: np.ndarray, n_assets: int) -> np.ndarray:
    """Valider que Sigma est une matrice carrée compatible avec n_assets."""
    Sigma = np.asarray(covariance_matrix, dtype=float)
    if Sigma.shape != (n_assets, n_assets):
        raise ValueError(
            f"Sigma doit être de forme ({n_assets}, {n_assets}), reçu {Sigma.shape}."
        )
    if not np.isfinite(Sigma).all():
        raise ValueError("Sigma contient NaN ou Inf.")
    return Sigma


# ============================================================
# 2. ESTIMATION DES PARAMÈTRES DE MARCHÉ
# ============================================================

def estimate_expected_returns(
    returns_matrix: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """
    Estimer le vecteur des rendements moyens journaliers.

    Formule :
        mu_hat = (1/T) * sum_t r_t

    Paramètres
    ----------
    returns_matrix : array-like, forme (T, n)
        Rendements journaliers des n actifs risqués.

    Retour
    ------
    ndarray, forme (n,)
        Moyenne empirique journalière de chaque actif.
    """
    R = _as_2d_returns(returns_matrix)
    if R.shape[0] < 2:
        raise ValueError("Il faut au moins 2 observations pour estimer mu.")
    return R.mean(axis=0)


def estimate_covariance_matrix(
    returns_matrix: np.ndarray | pd.DataFrame,
    shrinkage: float = 0.0,
) -> np.ndarray:
    """
    Estimer la matrice de covariance journalière des actifs risqués.

    La covariance empirique utilise le correcteur de Bessel (diviseur T-1).

    Régularisation optionnelle de type cible diagonale :
        Sigma_reg = (1 - alpha) * Sigma + alpha * (trace(Sigma)/n) * I

    Cette régularisation stabilise l'estimation lorsque T est proche de n,
    ou lorsque les actifs sont fortement corrélés. Elle est recommandée pour
    les mini-batchs SGD (faible T) afin d'éviter une matrice mal conditionnée.

    Paramètres
    ----------
    returns_matrix : array-like, forme (T, n)
    shrinkage : float dans [0, 1]
        0 = covariance empirique pure ; 1 = matrice diagonale uniforme.

    Retour
    ------
    ndarray, forme (n, n)
    """
    R = _as_2d_returns(returns_matrix)
    T, n = R.shape

    if T < 2:
        raise ValueError("Il faut au moins 2 observations pour estimer Sigma.")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage doit être dans [0, 1].")

    Sigma = np.cov(R, rowvar=False, ddof=1)
    # np.cov retourne un scalaire si n=1 ; on force une matrice 1×1.
    Sigma = np.atleast_2d(Sigma).astype(float)

    if shrinkage > 0.0:
        target_variance = np.trace(Sigma) / n
        Sigma = (1.0 - shrinkage) * Sigma + shrinkage * target_variance * np.eye(n)

    return Sigma


# ============================================================
# 3. CONTRAINTES ET PROJECTIONS
# ============================================================

def check_fully_invested_weights(
    weights: np.ndarray, tol: float = 1e-6
) -> bool:
    """
    Vérifier qu'un vecteur de poids est entièrement investi dans les actifs risqués.

    Condition :  sum(w) ≈ 1

    Utilisation : portefeuille classique sans poche cash (sum(w)=1 imposé).
    Pour le cas avec actif sans risque, utiliser check_risky_cash_weights().
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return False
    if not np.isfinite(w).all():
        return False
    return bool(np.isclose(w.sum(), 1.0, atol=tol))


# Alias rétro-compatible.
check_weights = check_fully_invested_weights


def check_risky_cash_weights(
    weights_risky: np.ndarray, tol: float = 1e-6
) -> bool:
    """
    Vérifier qu'un vecteur de poids risqués est compatible avec la présence
    d'un actif sans risque.

    Conditions long-only avec cash :
        w_i >= 0   pour tout i
        sum(w) <= 1

    Le poids sans risque résiduel est ensuite :
        w_rf = 1 - sum(w) >= 0
    """
    w = np.asarray(weights_risky, dtype=float)
    if w.size == 0:
        return False
    if not np.isfinite(w).all():
        return False
    if np.any(w < -tol):
        return False
    if w.sum() > 1.0 + tol:
        return False
    return True


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normaliser un vecteur non négatif pour que sa somme soit égale à 1.

    ATTENTION : cette fonction force un portefeuille entièrement investi dans
    les actifs risqués (w_rf = 0). Ne pas utiliser si l'on souhaite conserver
    une poche d'actif sans risque.
    """
    w = np.asarray(weights, dtype=float).copy()
    if np.any(w < -1e-12):
        raise ValueError(
            "normalize_weights reçoit des poids négatifs. Utiliser une projection adaptée."
        )
    s = float(w.sum())
    if np.isclose(s, 0.0):
        return np.ones(len(w)) / len(w)
    return w / s


def project_to_simplex(weights: np.ndarray) -> np.ndarray:
    """
    Projeter un vecteur sur le simplexe unitaire :

        Delta = {w >= 0, sum(w) = 1}

    Algorithme exact de Duchi et al. (2008), complexité O(n log n).

    Utilisation : portefeuille long-only entièrement investi dans les actifs
    risqués (sum(w)=1). Ne pas utiliser si l'on veut autoriser une poche cash.
    """
    w = np.asarray(weights, dtype=float).copy()
    if w.ndim != 1:
        raise ValueError("weights doit être un vecteur 1D.")
    if len(w) == 0:
        raise ValueError("weights ne peut pas être vide.")
    if not np.isfinite(w).all():
        raise ValueError("weights contient NaN ou Inf.")

    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    idx = np.arange(1, len(w) + 1)
    candidates = np.nonzero(u * idx > (cssv - 1.0))[0]
    if len(candidates) == 0:
        return np.ones(len(w)) / len(w)
    rho = candidates[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(w - theta, 0.0)


def project_to_nonnegative_weights(weights: np.ndarray) -> np.ndarray:
    """
    Clipper les poids négatifs puis renormaliser à somme 1.

    Cette projection est rapide mais n'est PAS la projection euclidienne exacte
    sur le simplexe. Elle force sum(w)=1 et élimine donc la poche cash.
    Préférer project_to_cash_simplex() dans le pipeline principal.
    """
    w = np.asarray(weights, dtype=float).copy()
    w = np.maximum(w, 0.0)
    return normalize_weights(w)


def project_to_cash_simplex(weights: np.ndarray) -> np.ndarray:
    """
    Projeter les poids risqués sur l'ensemble admissible avec cash :

        C = {w in R^n : w_i >= 0, sum(w) <= 1}

    Interprétation géométrique :
    - Si sum(w) <= 1 après clipping des négatifs : w est déjà admissible.
      Le résidu 1 - sum(w) est la poche cash.
    - Si sum(w) > 1 : le portefeuille est sur-investi. On ramène sur la
      face sum(w)=1 du simplexe par projection exacte (Duchi et al., 2008).

    C'est LA projection à utiliser dans le SGD lorsque l'actif sans risque
    doit pouvoir recevoir une allocation non nulle.
    """
    w = np.asarray(weights, dtype=float).copy()
    if w.ndim != 1:
        raise ValueError("weights doit être un vecteur 1D.")
    if len(w) == 0:
        raise ValueError("weights ne peut pas être vide.")
    if not np.isfinite(w).all():
        raise ValueError("weights contient NaN ou Inf.")

    # Interdit la vente à découvert.
    w = np.maximum(w, 0.0)

    # Si sur-investi, projeter sur la face sum(w)=1 du simplexe.
    if w.sum() > 1.0:
        w = project_to_simplex(w)

    return w


def enforce_portfolio_constraints(
    weights: np.ndarray,
    nonnegative: bool = True,
    exact_simplex: bool = False,
    allow_cash: bool = False,
) -> np.ndarray:
    """
    Appliquer les contraintes admissibles sur les poids risqués.

    Cas recommandé pour ce projet (avec actif sans risque) :
        nonnegative=True, allow_cash=True
    => projette sur {w >= 0, sum(w) <= 1}

    Cas Markowitz classique (sans cash) :
        nonnegative=True, allow_cash=False
    => projette sur {w >= 0, sum(w) = 1}

    Paramètres
    ----------
    weights : ndarray, forme (n,)
    nonnegative : bool
        Si True, interdit la vente à découvert (w_i >= 0).
    exact_simplex : bool
        Si True et allow_cash=False, utilise la projection exacte de Duchi
        sur {w >= 0, sum(w)=1}. Si False, utilise clip+renormalisation.
    allow_cash : bool
        Si True, projette sur {w >= 0, sum(w)<=1}.
        Si False, projette sur {w >= 0, sum(w)=1}.

    Retour
    ------
    ndarray, forme (n,)
    """
    w = np.asarray(weights, dtype=float).copy()

    if nonnegative and allow_cash:
        return project_to_cash_simplex(w)

    if nonnegative and not allow_cash:
        if exact_simplex:
            return project_to_simplex(w)
        return project_to_nonnegative_weights(w)

    # Cas long-short sans cash : renormalisation pour respecter sum(w)=1.
    s = float(w.sum())
    if np.isclose(s, 0.0):
        return np.ones(len(w)) / len(w)
    return w / s


# ============================================================
# 4. ACTIF SANS RISQUE
# ============================================================

def compute_risk_free_weight(
    weights_risky: np.ndarray, tol: float = 1e-6
) -> float:
    """
    Calculer le poids de l'actif sans risque comme résidu budgétaire :

        w_rf = 1 - sum(w_risky)

    Interprétation :
      w_rf > 0  : une partie du capital est en actif sans risque.
      w_rf = 0  : tout le capital est en actifs risqués.
      w_rf < 0  : portefeuille sur-investi (levier), interdit dans ce projet.

    Lève une ValueError si w_rf < -tol.
    """
    w = np.asarray(weights_risky, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights_risky doit être un vecteur 1D.")
    if not np.isfinite(w).all():
        raise ValueError("weights_risky contient NaN ou Inf.")

    w_rf = 1.0 - float(w.sum())

    if w_rf < -tol:
        raise ValueError(
            f"Portefeuille sur-investi : sum(w)={float(w.sum()):.6f}, "
            f"donc w_rf={w_rf:.6f} < 0. "
            "Utiliser project_to_cash_simplex() ou "
            "enforce_portfolio_constraints(..., allow_cash=True) avant d'appeler cette fonction."
        )

    # Supprime un -0.0 résiduel dû aux erreurs d'arrondi.
    if abs(w_rf) <= tol:
        return 0.0
    return w_rf


def split_weights(
    weights: np.ndarray, tol: float = 1e-6
) -> tuple[np.ndarray, float]:
    """
    Séparer un vecteur complet [w_1, ..., w_n, w_rf] en poids risqués et
    poids sans risque.

    Convention : le DERNIER élément du vecteur est le poids de l'actif sans risque.

    Exemple :
        weights = [0.3, 0.4, 0.2, 0.1]  =>  w_risky=[0.3,0.4,0.2], w_rf=0.1
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) < 2:
        raise ValueError(
            "weights doit être un vecteur 1D contenant au moins 2 éléments."
        )

    w_risky = w[:-1]
    w_rf = float(w[-1])
    budget = float(w_risky.sum()) + w_rf

    if abs(budget - 1.0) > tol:
        raise ValueError(
            f"Contrainte budgétaire violée : sum(w)+w_rf={budget:.6f} ≠ 1."
        )

    return w_risky, w_rf


# ============================================================
# 5. RENDEMENT, RISQUE ET SHARPE
# ============================================================

def portfolio_expected_return(
    weights: np.ndarray, expected_returns: np.ndarray
) -> float:
    """
    Calculer le rendement espéré journalier de la poche risquée :

        E[R_risky] = w^T mu

    Cette fonction n'inclut PAS l'actif sans risque.
    Pour le rendement total (risqué + cash), utiliser
    portfolio_expected_return_with_cash().
    """
    w, mu = _validate_weights_and_mu(weights, expected_returns)
    return float(w @ mu)


def portfolio_expected_return_with_cash(
    weights_risky: np.ndarray,
    expected_returns: np.ndarray,
    w_rf: float | None = None,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    annualized: bool = True,
) -> float:
    """
    Calculer le rendement espéré total du portefeuille (risqué + cash).

    Version annualisée (annualized=True, par défaut) :

        mu_p_ann = T * w^T mu_daily + w_rf * r_f_annual

    Homogénéité dimensionnelle :
        T * w^T mu_daily  => rendement annuel de la poche risquée
        w_rf * r_f_annual => rendement annuel de la poche cash
        Les deux termes sont bien homogènes.

    Version journalière (annualized=False) :

        mu_p_daily = w^T mu_daily + w_rf * r_f_daily
        avec r_f_daily = (1 + r_f_annual)^(1/T) - 1

    Paramètres
    ----------
    weights_risky : ndarray, forme (n,)
    expected_returns : ndarray, forme (n,)
        Rendements moyens JOURNALIERS des actifs risqués.
    w_rf : float ou None
        Si None, calculé comme 1 - sum(weights_risky).
    risk_free_rate : float
        Taux sans risque ANNUEL.
    annualization_factor : int
        Nombre de jours de bourse par an (T = 252).
    annualized : bool
        True => rendement annualisé ; False => rendement journalier.
    """
    w, mu = _validate_weights_and_mu(weights_risky, expected_returns)

    if w_rf is None:
        w_rf = compute_risk_free_weight(w)
    else:
        budget = float(w.sum()) + float(w_rf)
        if abs(budget - 1.0) > 1e-6:
            raise ValueError(
                f"Contrainte budgétaire violée : sum(w)+w_rf={budget:.6f} ≠ 1."
            )

    risky_daily = float(w @ mu)

    if annualized:
        # T * mu_daily est homogène à r_f_annual (tous deux en rendement annuel).
        return annualization_factor * risky_daily + float(w_rf) * risk_free_rate

    rf_daily = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0
    return risky_daily + float(w_rf) * rf_daily


def portfolio_variance(
    weights: np.ndarray, covariance_matrix: np.ndarray
) -> float:
    """
    Calculer la variance journalière du portefeuille risqué :

        Var(R_p) = w^T Sigma w

    L'actif sans risque ne contribue pas à la variance (variance nulle,
    covariance nulle avec les actifs risqués).
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights doit être un vecteur 1D.")
    Sigma = _validate_covariance(covariance_matrix, len(w))
    return float(w @ Sigma @ w)


def portfolio_volatility(
    weights: np.ndarray, covariance_matrix: np.ndarray
) -> float:
    """Calculer la volatilité journalière : sqrt(w^T Sigma w)."""
    return float(np.sqrt(max(portfolio_variance(weights, covariance_matrix), 0.0)))


def portfolio_sharpe(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    include_cash: bool = False,
) -> float:
    """
    Calculer le ratio de Sharpe EX-ANTE annualisé.

    Ce ratio est calculé à partir des paramètres estimés (mu, Sigma), PAS
    des rendements réalisés. Il diffère du Sharpe ex-post du backtest.

    Si include_cash=False (portefeuille entièrement risqué) :
        SR = (T * w^T mu - r_f) / (sqrt(T) * sqrt(w^T Sigma w))

    Si include_cash=True (avec poche cash résiduelle) :
        w_rf = 1 - sum(w)
        mu_p_ann = T * w^T mu + w_rf * r_f
        sigma_p_ann = sqrt(T) * sqrt(w^T Sigma w)
        SR = (mu_p_ann - r_f) / sigma_p_ann

    Remarque : si le portefeuille est 100 % cash, sigma_p_ann = 0 et le
    Sharpe est retourné à 0.0 par convention.
    """
    w, mu = _validate_weights_and_mu(weights, expected_returns)
    Sigma = _validate_covariance(covariance_matrix, len(w))

    if include_cash:
        mu_ann = portfolio_expected_return_with_cash(
            w,
            mu,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
            annualized=True,
        )
    else:
        mu_ann = annualization_factor * float(w @ mu)

    sigma_ann = np.sqrt(annualization_factor) * np.sqrt(
        max(float(w @ Sigma @ w), 0.0)
    )
    if np.isclose(sigma_ann, 0.0):
        return 0.0
    return float((mu_ann - risk_free_rate) / sigma_ann)


# ============================================================
# 6. FONCTION DE COÛT MEAN-VARIANCE SANS CASH EFFECTIF
# ============================================================

def mean_variance_cost(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    lambda_risk: float = 1.0,
    annualization_factor: int = 252,
) -> float:
    """
    Fonction de coût mean-variance annualisée, SANS actif sans risque.

        J(w) = lambda * T * w^T Sigma w - T * w^T mu

    À utiliser uniquement si sum(w) = 1 (portefeuille entièrement risqué).
    Pour le projet avec actif sans risque, préférer mean_variance_cost_with_cash().

    Paramètres
    ----------
    lambda_risk : float >= 0
        Paramètre d'aversion au risque. Plus il est grand, plus on pénalise
        la variance par rapport au rendement.
    annualization_factor : int
        T = 252 (nombre de jours de bourse par an).
    """
    if lambda_risk < 0:
        raise ValueError("lambda_risk doit être positif ou nul.")

    w, mu = _validate_weights_and_mu(weights, expected_returns)
    Sigma = _validate_covariance(covariance_matrix, len(w))
    T = annualization_factor

    return float(lambda_risk * T * (w @ Sigma @ w) - T * (w @ mu))


def mean_variance_gradient(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    lambda_risk: float = 1.0,
    annualization_factor: int = 252,
) -> np.ndarray:
    """
    Gradient analytique de mean_variance_cost par rapport à w.

    Dérivation :
        J(w) = lambda*T * w^T Sigma w - T * w^T mu
        dJ/dw = T * (2*lambda*Sigma*w - mu)

    Paramètres
    ----------
    (identiques à mean_variance_cost)

    Retour
    ------
    ndarray, forme (n,)
    """
    if lambda_risk < 0:
        raise ValueError("lambda_risk doit être positif ou nul.")

    w, mu = _validate_weights_and_mu(weights, expected_returns)
    Sigma = _validate_covariance(covariance_matrix, len(w))
    T = annualization_factor

    return T * (2.0 * lambda_risk * Sigma @ w - mu)


# ============================================================
# 7. FONCTION DE COÛT MEAN-VARIANCE AVEC CASH (VERSION PRINCIPALE)
# ============================================================

def mean_variance_cost_with_cash(
    weights_risky: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float,
    lambda_risk: float = 1.0,
    annualization_factor: int = 252,
) -> float:
    """
    Fonction de coût mean-variance avec actif sans risque comme résidu.

    C'est la fonction objective principale du projet.

    Notation :
        mu        : rendements JOURNALIERS des actifs risqués
        r_f       : taux sans risque ANNUEL
        T         : annualization_factor = 252
        w_rf      : 1 - sum(w)

    Définition (tous les termes sont en rendement annuel) :

        J(w) = lambda * T * w^T Sigma w
               - [T * w^T mu + (1 - sum(w)) * r_f]

    Homogénéité dimensionnelle :
        T * w^T Sigma w : variance annualisée (Sigma journalière * T)
        T * w^T mu      : rendement annualisé de la poche risquée
        (1-sum(w))*r_f  : rendement annuel de la poche cash (r_f déjà annuel)
        => tous les termes sont en rendement annuel. ✓

    Interprétation économique :
        Minimiser J revient à maximiser le rendement annuel total moins
        lambda fois la variance annualisée, ce qui correspond à un investisseur
        avec une aversion au risque de coefficient lambda.
    """
    if lambda_risk < 0:
        raise ValueError("lambda_risk doit être positif ou nul.")

    w, mu = _validate_weights_and_mu(weights_risky, expected_returns)
    Sigma = _validate_covariance(covariance_matrix, len(w))
    w_rf = compute_risk_free_weight(w)
    T = annualization_factor

    risky_return_ann  = T * float(w @ mu)        # rendement annualisé, poche risquée
    cash_return_ann   = w_rf * risk_free_rate     # rendement annuel, poche cash
    risky_variance_ann = T * float(w @ Sigma @ w) # variance annualisée

    return float(lambda_risk * risky_variance_ann - (risky_return_ann + cash_return_ann))


def mean_variance_gradient_with_cash(
    weights_risky: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float,
    lambda_risk: float = 1.0,
    annualization_factor: int = 252,
) -> np.ndarray:
    """
    Gradient analytique de mean_variance_cost_with_cash par rapport à w.

    Dérivation terme par terme :
        d/dw [lambda*T*w^T Sigma w]   = T * 2*lambda*Sigma*w
        d/dw [-T * w^T mu]             = -T * mu
        d/dw [-(1 - sum(w)) * r_f]    = +r_f * 1_n
          (car d/dw_i [r_f * sum_j w_j] = r_f)

    Gradient total :
        grad J_cash(w) = T * (2*lambda*Sigma*w - mu) + r_f * 1_n

    Le terme +r_f * 1_n est le signal économique clé : il pénalise les
    allocations qui réduisent la poche cash, forçant le SGD à n'investir
    dans les actifs risqués que si leur rendement excède r_f.

    Retour
    ------
    ndarray, forme (n,)
    """
    if lambda_risk < 0:
        raise ValueError("lambda_risk doit être positif ou nul.")

    w, mu = _validate_weights_and_mu(weights_risky, expected_returns)
    Sigma = _validate_covariance(covariance_matrix, len(w))
    T = annualization_factor

    return T * (2.0 * lambda_risk * Sigma @ w - mu) + risk_free_rate * np.ones_like(w)


# ============================================================
# 8. RÉFÉRENCE ANALYTIQUE : FRONTIÈRE EFFICIENTE (MARKOWITZ)
# ============================================================

def markowitz_analytical_weights(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    lambda_risk: float = 1.0,
    nonnegative: bool = False,
) -> np.ndarray:
    """
    Calculer les poids optimaux analytiques du problème mean-variance SANS cash.

    Ce problème admet une solution en forme fermée par inversion de Sigma
    lorsque la contrainte long-only n'est PAS imposée :

        w* = argmin_{sum(w)=1}  lambda * w^T Sigma w  -  w^T mu

    Remarque sur annualization_factor
    ----------------------------------
    Le facteur T (annualization_factor) factorise TOUTE la fonction objectif :

        J(w) = T * (lambda * w^T Sigma w - w^T mu)

    Il disparaît donc dans les conditions de premier ordre (dL/dw = 0) et
    n'influence PAS la solution analytique w*. Ce paramètre a été retiré de
    la signature pour éviter toute confusion. Si vous utilisez cette fonction
    comme référence pour valider la convergence du SGD, comparez les poids,
    pas les valeurs de la fonction objectif (qui dépendent de T).

    Dérivation par multiplicateurs de Lagrange
    ------------------------------------------
        L(w, nu) = lambda * w^T Sigma w - w^T mu + nu * (sum(w) - 1)

        dL/dw = 0  =>  2*lambda*Sigma*w - mu + nu*1 = 0
                   =>  w = (1/(2*lambda)) * Sigma^{-1} * (mu - nu*1)

    En injectant dans sum(w) = 1, on obtient :

        nu = (1^T Sigma^{-1} mu - 2*lambda) / (1^T Sigma^{-1} 1)

    puis :

        w* = (1/(2*lambda)) * Sigma^{-1} * (mu - nu*1)

    Utilisation dans le rapport :
        Ces poids servent de RÉFÉRENCE pour valider la convergence du SGD.
        Si nonnegative=True, le problème n'a pas de solution analytique simple
        et on utilise une projection post-hoc (heuristique, non-optimale).

    Paramètres
    ----------
    expected_returns : ndarray, forme (n,)
        Rendements des actifs risqués (journaliers ou annualisés, peu importe :
        T factorise l'objectif et ne change pas w*).
    covariance_matrix : ndarray, forme (n, n)
        Matrice de covariance (même fréquence que expected_returns).
    lambda_risk : float > 0
        Coefficient d'aversion au risque.
    nonnegative : bool
        Si True, projette les poids sur le simplexe après optimisation.
        Cela fournit une approximation, PAS la solution exacte du problème
        contraint. Utiliser scipy/CVXPY pour la solution exacte long-only.

    Retour
    ------
    ndarray, forme (n,)
    """
    if lambda_risk <= 0:
        raise ValueError("lambda_risk doit être strictement positif.")

    mu = np.asarray(expected_returns, dtype=float)
    Sigma = _validate_covariance(covariance_matrix, len(mu))
    n = len(mu)

    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Sigma est singulière. Utiliser estimate_covariance_matrix avec shrinkage > 0."
        ) from exc

    ones = np.ones(n)
    # Résolution par multiplicateurs de Lagrange (sans contrainte long-only).
    A = Sigma_inv @ mu          # Sigma^{-1} mu
    B = Sigma_inv @ ones        # Sigma^{-1} 1
    denom = 2.0 * lambda_risk
    # Multiplicateur de Lagrange nu (T a été simplifié dans dL/dw = 0).
    nu = (ones @ A - denom) / (ones @ B)
    w_star = (A - nu * B) / denom

    if nonnegative:
        # Approximation post-hoc : projection sur le simplexe.
        # Ce n'est PAS la solution optimale du problème contraint.
        w_star = project_to_simplex(w_star)

    return w_star