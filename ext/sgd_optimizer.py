"""
sgd_optimizer.py
================
Optimisation du portefeuille par descente de gradient stochastique (SGD).

Fondements théoriques
---------------------
On minimise la fonction de coût mean-variance annualisée :

    J(w) = λ · T · wᵀΣw  −  T · wᵀµ

par mini-batch SGD avec projection à chaque itération.

Algorithme (itération k)
------------------------
1. Tirer aléatoirement un mini-batch B_k ⊂ {1,...,T}, |B_k| = b
2. Estimer µ̂_k et Σ̂_k sur B_k  (estimateurs sans biais)
3. Calculer le gradient stochastique :
       g_k = T · (2λ Σ̂_k w_k − µ̂_k)
4. Mise à jour SGD :  w_{k+1} ← w_k − η · g_k
5. Projection :       w_{k+1} ← Π_Δ(w_{k+1})
   où Δ = {w ≥ 0, 1ᵀw = 1} si long-only, sinon {1ᵀw = 1}

Convergence (théorie)
---------------------
Pour η suffisamment petit et J convexe (Σ ≻ 0), SGD converge
en espérance vers le minimum global. En pratique on utilise
un critère d'arrêt basé sur la variation de J sur une fenêtre.

Gradient numérique (optionnel)
------------------------------
Pour toute fonction de coût f arbitraire, on approche le gradient
par différences finies centrées :

    ∂f/∂w_i ≈ [f(w + εe_i) − f(w − εe_i)] / (2ε)

C'est O(n) appels à f par itération, à n'utiliser que si le
gradient analytique n'est pas disponible.
"""

import numpy as np
from ext.portfolio_math import (
    mean_variance_gradient,
    enforce_portfolio_constraints,
    check_weights,
)


# ============================================================
# 1. INITIALISATION DES POIDS
# ============================================================

def initialize_weights(n_assets: int, method: str = "equal") -> np.ndarray:
    """
    Initialiser le vecteur de poids w⁰ ∈ Δ.

    Méthodes disponibles
    --------------------
    "equal"  : w_i = 1/n  pour tout i  (équipondération)
    "random" : w ~ Dir(1,...,1)  puis normalisé sur Δ

    L'initialisation équipondérée est recommandée car elle
    correspond au minimum de la variance dans le cas isotrope
    (Σ = σ²I), ce qui accélère la convergence.

    Paramètres
    ----------
    n_assets : int > 0
    method : str

    Retour
    ------
    w : (n,) ndarray, avec w ≥ 0 et 1ᵀw = 1
    """
    if n_assets <= 0:
        raise ValueError("n_assets doit être un entier strictement positif.")

    if method == "equal":
        return np.ones(n_assets) / n_assets

    if method == "random":
        w = np.random.dirichlet(np.ones(n_assets))   # distribution uniforme sur Δ
        return w

    raise ValueError(f"Méthode d'initialisation inconnue : '{method}'.")


# ============================================================
# 2. GRADIENT NUMÉRIQUE (DIFFÉRENCES FINIES CENTRÉES)
# ============================================================

def approximate_gradient(
    cost_function,
    weights: np.ndarray,
    epsilon: float = 1e-5,
    **kwargs
) -> np.ndarray:
    """
    Approximer ∇_w f par différences finies centrées.

    Formule (ordre 2 en ε) :
        ∂f/∂w_i ≈ [f(w + εe_i) − f(w − εe_i)] / (2ε)

    L'ordre 2 réduit l'erreur de troncature à O(ε²) contre O(ε)
    pour la méthode forward-only. Le choix ε ≈ 1e-5 est un bon
    compromis entre erreur de troncature et erreur d'arrondi flottant.

    Complexité : 2n appels à f.

    Paramètres
    ----------
    cost_function : callable  f(w, **kwargs) → float
    weights : (n,) ndarray
    epsilon : float > 0
    **kwargs : arguments transmis à cost_function

    Retour
    ------
    grad : (n,) ndarray
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    grad = np.zeros(n)

    for i in range(n):
        w_plus  = w.copy(); w_plus[i]  += epsilon
        w_minus = w.copy(); w_minus[i] -= epsilon
        grad[i] = (cost_function(w_plus, **kwargs) - cost_function(w_minus, **kwargs)) / (2.0 * epsilon)

    return grad


# ============================================================
# 3. MISE À JOUR SGD
# ============================================================

def sgd_update(
    weights: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float
) -> np.ndarray:
    """
    Effectuer un pas de gradient :

        w ← w − η · g

    Paramètres
    ----------
    weights : (n,)
    gradient : (n,)
    learning_rate : float > 0 — pas η

    Retour
    ------
    w_new : (n,)
    """
    w = np.asarray(weights, dtype=float)
    g = np.asarray(gradient, dtype=float)
    if w.shape != g.shape:
        raise ValueError("weights et gradient doivent avoir la même forme.")
    if learning_rate <= 0:
        raise ValueError("learning_rate doit être strictement positif.")
    return w - learning_rate * g


# ============================================================
# 4. APPLICATION DES CONTRAINTES
# ============================================================

def apply_constraints(
    weights: np.ndarray,
    constraint_function=None,
    nonnegative: bool = True,
    exact_simplex: bool = True
) -> np.ndarray:
    """
    Projeter les poids sur l'ensemble admissible après une mise à jour.

    Si constraint_function est fourni, il remplace la projection par défaut.
    Sinon on utilise enforce_portfolio_constraints de portfolio_math.

    Paramètres
    ----------
    weights : (n,)
    constraint_function : callable or None
    nonnegative : bool
    exact_simplex : bool — projection exacte L² si True

    Retour
    ------
    w_proj : (n,)
    """
    w = np.asarray(weights, dtype=float)
    if constraint_function is not None:
        if not callable(constraint_function):
            raise TypeError("constraint_function doit être callable.")
        return np.asarray(constraint_function(w), dtype=float)
    return enforce_portfolio_constraints(w, nonnegative=nonnegative, exact_simplex=exact_simplex)


# ============================================================
# 5. CRITÈRE DE CONVERGENCE
# ============================================================

def check_convergence(
    cost_history: list,
    tolerance: float = 1e-6,
    window: int = 10
) -> bool:
    """
    Détecter la convergence par stabilité de la fonction de coût.

    Critère : max(J_{k-w}, ..., J_k) − min(J_{k-w}, ..., J_k) < tolerance

    Ce critère mesure la variation totale de J sur la fenêtre glissante.
    Il est robuste au bruit stochastique car il ne compare pas
    deux valeurs consécutives.

    Paramètres
    ----------
    cost_history : list of float — historique de J
    tolerance : float — seuil de variation minimale
    window : int — taille de la fenêtre d'observation

    Retour
    ------
    converged : bool
    """
    if len(cost_history) < window:
        return False
    recent = cost_history[-window:]
    return (max(recent) - min(recent)) < tolerance


# ============================================================
# 6. ALGORITHME SGD PRINCIPAL
# ============================================================

def run_sgd(
    returns_matrix: np.ndarray,
    initial_weights: np.ndarray,
    n_iterations: int,
    learning_rate: float,
    lambda_risk: float = 1.0,
    batch_size: int = 64,
    constraint_function=None,
    nonnegative: bool = True,
    exact_simplex: bool = True,
    annualization_factor: int = 252,
    convergence_tolerance: float = 1e-6,
    convergence_window: int = 10,
    verbose: bool = False
) -> dict:
    """
    Exécuter l'optimisation SGD du portefeuille.

    Algorithme (itération k)
    ------------------------
    1. Mini-batch B_k  de taille min(b, T) tiré sans remise
    2. µ̂_k = mean(B_k),  Σ̂_k = cov(B_k)     (estimateurs sur le batch)
    3. g_k  = T · (2λ Σ̂_k w_k − µ̂_k)        (gradient analytique)
    4. w_{k+1} = w_k − η · g_k               (pas SGD)
    5. w_{k+1} = Π_Δ(w_{k+1})                (projection)
    6. J_k = λT · w_k Σ̂_k w_k − T · w_k µ̂_k (coût)
    7. Critère d'arrêt anticipé si convergence détectée

    Note : µ̂_k et Σ̂_k sont calculés sur B_k uniquement,
    ce qui constitue le caractère « stochastique » de l'algorithme.
    Le gradient g_k est donc un estimateur sans biais de ∇J
    (en espérance sur le tirage aléatoire du batch).

    Paramètres
    ----------
    returns_matrix : (T, n) array — rendements historiques
    initial_weights : (n,) — poids initiaux
    n_iterations : int — nombre maximal d'itérations
    learning_rate : float > 0 — pas η
    lambda_risk : float > 0 — aversion au risque
    batch_size : int — taille du mini-batch b (clipé à T si b > T)
    constraint_function : callable or None — contrainte personnalisée
    nonnegative : bool — long-only si True
    exact_simplex : bool — projection exacte L² sur Δ
    annualization_factor : int — T = 252
    convergence_tolerance : float — seuil de convergence
    convergence_window : int — fenêtre de convergence
    verbose : bool — affichage périodique

    Retour
    ------
    dict avec :
      "final_weights"   : (n,) — poids optimisés
      "cost_history"    : list[float] — J à chaque itération
      "weights_history" : list[ndarray] — w à chaque itération
      "n_iterations"    : int — itérations réellement effectuées
      "converged"       : bool
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    T, n = R.shape

    w = np.asarray(initial_weights, dtype=float).copy()
    if len(w) != n:
        raise ValueError(f"initial_weights a {len(w)} éléments mais returns_matrix a {n} colonnes.")

    # Clip du batch_size pour éviter l'erreur replace=False
    b = min(batch_size, T)

    cost_history: list[float] = []
    weights_history: list[np.ndarray] = []
    converged = False
    k = 0

    for k in range(n_iterations):
        # --- 1. Tirage du mini-batch sans remise ---
        idx   = np.random.choice(T, size=b, replace=False)
        batch = R[idx]                              # (b, n)

        # --- 2. Estimation statistique sur le batch ---
        mu_batch    = batch.mean(axis=0)            # (n,)  estimateur de µ
        if b < 2:
            Sigma_batch = np.zeros((n, n))          # cas dégénéré
        else:
            Sigma_batch = np.cov(batch, rowvar=False)  # (n,n) Bessel-corrigé

        # --- 3. Gradient analytique stochastique ---
        g = mean_variance_gradient(
            weights=w,
            expected_returns=mu_batch,
            covariance_matrix=Sigma_batch,
            lambda_risk=lambda_risk,
            annualization_factor=annualization_factor
        )

        # --- 4. Mise à jour SGD ---
        w = sgd_update(w, g, learning_rate)

        # --- 5. Projection sur l'ensemble admissible ---
        w = apply_constraints(
            w,
            constraint_function=constraint_function,
            nonnegative=nonnegative,
            exact_simplex=exact_simplex
        )

        # --- 6. Calcul du coût (sur le batch) ---
        cost = (
            lambda_risk * annualization_factor * float(w @ Sigma_batch @ w)
            - annualization_factor * float(w @ mu_batch)
        )
        cost_history.append(cost)
        weights_history.append(w.copy())

        # --- 7. Critère d'arrêt anticipé ---
        if check_convergence(cost_history, convergence_tolerance, convergence_window):
            converged = True
            if verbose:
                print(f"Convergence détectée à l'itération {k} — J = {cost:.6f}")
            break

        if verbose and k % 50 == 0:
            print(f"Itération {k:>5d}  J = {cost:+.6f}  ||g|| = {np.linalg.norm(g):.4e}")

    return {
        "final_weights":   w,
        "cost_history":    cost_history,
        "weights_history": weights_history,
        "n_iterations":    k + 1,
        "converged":       converged,
    }