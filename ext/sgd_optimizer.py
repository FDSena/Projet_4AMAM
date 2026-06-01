"""
sgd_optimizer.py
================
Optimisation d'un portefeuille par descente de gradient stochastique (SGD).

Ce module est cohérent avec portfolio_math.py : les données de marché
contiennent uniquement les actifs risqués. L'actif sans risque n'est pas une
colonne de la matrice de rendements ; il est calculé comme résidu budgétaire :

    w_rf = 1 - sum_i w_i

Justification mathématique du SGD dans ce contexte
----------------------------------------------------
La fonction de coût mean-variance est déterministe une fois mu et Sigma fixés.
Cependant, on ne dispose que d'estimations empiriques de ces paramètres à
partir des données historiques.

L'algorithme implémenté ici est un **SGD avec ré-estimation stochastique des
paramètres de marché** : à chaque itération, mu_batch et Sigma_batch sont
ré-estimés sur un sous-ensemble tiré aléatoirement (mini-batch). Cela introduit
un bruit artificiel dans le gradient, ce qui présente deux avantages :

  1. Robustesse à l'estimation : les poids finaux ne sur-ajustent pas une
     estimation unique de mu et Sigma calculée sur l'historique complet.
  2. Exploration : le bruit permet d'échapper à des optima locaux plats
     (bien que la fonction soit convexe en w, les estimateurs bruités
     peuvent créer des paysages non convexes effectifs).

Ce n'est PAS un SGD au sens strict de l'optimisation stochastique (où la
fonction objectif est une somme de pertes individuelles). Il doit être décrit
dans le rapport comme "SGD avec ré-estimation stochastique des paramètres".

Objectif recommandé avec actif sans risque
------------------------------------------
Les rendements des actifs risqués sont journaliers, le taux sans risque est
annuel, et T = annualization_factor = 252.

    J_cash(w) = lambda * T * w^T Sigma w
                - [T * w^T mu + (1 - sum(w)) * r_f]

Gradient :

    grad J_cash(w) = T * (2*lambda*Sigma*w - mu) + r_f * 1_n

Algorithme SGD projeté
-----------------------
À l'itération k :

  1. Tirer un mini-batch de taille B sans remise.
  2. Estimer mu_batch et Sigma_batch sur ce mini-batch (avec régularisation).
  3. Calculer le gradient stochastique sur (mu_batch, Sigma_batch).
  4. Mettre à jour : w <- w - eta_k * grad.
  5. Projeter w sur l'ensemble admissible (voir apply_constraints).
  6. Sauvegarder l'historique des coûts et des poids.

Schedule du taux d'apprentissage
---------------------------------
Un taux fixe ne garantit pas la convergence vers le minimum : le SGD oscille
autour de la solution dans un rayon proportionnel à eta * sigma_grad.
Trois schedules sont disponibles :

  "constant"   : eta_k = eta_0         (rapide, peut osciller)
  "inverse"    : eta_k = eta_0 / (1+k) (convergence théorique garantie
                                         sous hypothèses standard)
  "sqrt"       : eta_k = eta_0 / sqrt(1+k) (compromis vitesse/convergence)

Le schedule "inverse" satisfait les conditions de Robbins-Monro :
    sum_k eta_k = inf  et  sum_k eta_k^2 < inf
ce qui garantit la convergence vers l'optimum pour des fonctions convexes.
"""

from __future__ import annotations

import numpy as np

from ext.portfolio_math import (
    estimate_covariance_matrix,
    mean_variance_cost,
    mean_variance_gradient,
    mean_variance_cost_with_cash,
    mean_variance_gradient_with_cash,
    enforce_portfolio_constraints,
    compute_risk_free_weight,
)


# ============================================================
# 1. INITIALISATION DES POIDS
# ============================================================

def initialize_weights(
    n_assets: int,
    method: str = "equal",
    risky_budget: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Initialiser les poids investis dans les actifs risqués.

    Paramètres
    ----------
    n_assets : int
        Nombre d'actifs risqués.
    method : {"equal", "random", "cash"}
        - "equal"  : répartition égale du budget risqué (w_i = risky_budget/n).
        - "random" : répartition aléatoire Dirichlet du budget risqué.
        - "cash"   : aucun investissement risqué (w = 0, w_rf = 1).
    risky_budget : float dans [0, 1]
        Fraction initialement investie dans les actifs risqués.
        La fraction restante (1 - risky_budget) est en actif sans risque.
    random_state : int ou None
        Graine pour la reproductibilité.

    Retour
    ------
    ndarray, forme (n_assets,)
        Vecteur w avec w_i >= 0 et sum(w) = risky_budget <= 1.
    """
    if n_assets <= 0:
        raise ValueError("n_assets doit être strictement positif.")
    if not 0.0 <= risky_budget <= 1.0:
        raise ValueError("risky_budget doit appartenir à [0, 1].")

    if method == "cash":
        return np.zeros(n_assets)

    if method == "equal":
        return np.ones(n_assets) * (risky_budget / n_assets)

    if method == "random":
        rng = np.random.default_rng(random_state)
        return risky_budget * rng.dirichlet(np.ones(n_assets))

    raise ValueError("method doit être 'equal', 'random' ou 'cash'.")


# ============================================================
# 2. SCHEDULE DU TAUX D'APPRENTISSAGE
# ============================================================

def compute_learning_rate(
    base_lr: float,
    iteration: int,
    schedule: str = "constant",
) -> float:
    """
    Calculer le taux d'apprentissage à l'itération k.

    Schedules disponibles
    ---------------------
    "constant"  : eta_k = eta_0
    "inverse"   : eta_k = eta_0 / (1 + k)
        Satisfait les conditions de Robbins-Monro.
        Convergence vers l'optimum garantie pour fonctions convexes.
    "sqrt"      : eta_k = eta_0 / sqrt(1 + k)
        Compromis entre vitesse initiale et convergence asymptotique.

    Paramètres
    ----------
    base_lr : float > 0
        Taux initial eta_0.
    iteration : int >= 0
        Numéro de l'itération courante.
    schedule : str
        Nom du schedule.

    Retour
    ------
    float
        Taux d'apprentissage effectif à l'itération k.
    """
    if base_lr <= 0:
        raise ValueError("base_lr doit être strictement positif.")
    if iteration < 0:
        raise ValueError("iteration doit être >= 0.")

    if schedule == "constant":
        return base_lr
    if schedule == "inverse":
        return base_lr / (1.0 + iteration)
    if schedule == "sqrt":
        return base_lr / np.sqrt(1.0 + iteration)

    raise ValueError("schedule doit être 'constant', 'inverse' ou 'sqrt'.")


# ============================================================
# 3. GRADIENT NUMÉRIQUE (VÉRIFICATION)
# ============================================================

def approximate_gradient(
    cost_function,
    weights: np.ndarray,
    epsilon: float = 1e-5,
    **kwargs,
) -> np.ndarray:
    """
    Approximer le gradient par différences finies centrées.

    Formule :
        df/dw_i ≈ [f(w + epsilon*e_i) - f(w - epsilon*e_i)] / (2*epsilon)

    Utilisation : vérification du gradient analytique (test de cohérence),
    ou utilisation d'une fonction de coût personnalisée sans gradient analytique.
    Pour la fonction mean-variance standard, préférer le gradient analytique.

    ATTENTION : si cost_function dépend de mu et Sigma via kwargs, cette
    fonction ignore les statistiques du batch et n'est donc PAS stochastique.
    Elle doit être utilisée avec des paramètres fixés, pas dans la boucle SGD.
    """
    if epsilon <= 0:
        raise ValueError("epsilon doit être strictement positif.")

    w = np.asarray(weights, dtype=float)
    grad = np.zeros_like(w)

    for i in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[i] += epsilon
        w_minus[i] -= epsilon
        grad[i] = (
            cost_function(w_plus, **kwargs) - cost_function(w_minus, **kwargs)
        ) / (2.0 * epsilon)

    return grad


# ============================================================
# 4. MISE À JOUR SGD
# ============================================================

def sgd_update(
    weights: np.ndarray, gradient: np.ndarray, learning_rate: float
) -> np.ndarray:
    """
    Effectuer un pas de descente de gradient projeté :

        w_new = w - eta * grad

    La projection sur l'ensemble admissible est effectuée séparément par
    apply_constraints(), conformément à l'algorithme SGD projeté.
    """
    w = np.asarray(weights, dtype=float)
    g = np.asarray(gradient, dtype=float)

    if w.shape != g.shape:
        raise ValueError("weights et gradient doivent avoir la même forme.")
    if learning_rate <= 0:
        raise ValueError("learning_rate doit être strictement positif.")

    return w - learning_rate * g


# ============================================================
# 5. CONTRAINTES DE PORTEFEUILLE
# ============================================================

def apply_constraints(
    weights: np.ndarray,
    constraint_function=None,
    nonnegative: bool = True,
    allow_cash: bool = True,
    exact_simplex: bool = True,
) -> np.ndarray:
    """
    Appliquer les contraintes après une mise à jour SGD.

    Cas recommandé pour ce projet (avec actif sans risque) :
        nonnegative=True, allow_cash=True
    => projette sur {w >= 0, sum(w) <= 1}

    Cas Markowitz classique (sans cash) :
        nonnegative=True, allow_cash=False
    => projette sur {w >= 0, sum(w) = 1}

    Paramètres
    ----------
    constraint_function : callable ou None
        Fonction personnalisée de projection : w -> w_projected.
        Si fournie, elle remplace la projection par défaut.
        ATTENTION : cette fonction doit être déterministe et idempotente.
    nonnegative : bool
    allow_cash : bool
    exact_simplex : bool
        Si True et allow_cash=False, utilise l'algorithme exact de Duchi.
    """
    w = np.asarray(weights, dtype=float)

    if constraint_function is not None:
        if not callable(constraint_function):
            raise TypeError("constraint_function doit être callable.")
        return np.asarray(constraint_function(w), dtype=float)

    return enforce_portfolio_constraints(
        w,
        nonnegative=nonnegative,
        allow_cash=allow_cash,
        exact_simplex=exact_simplex,
    )


# ============================================================
# 6. CONVERGENCE
# ============================================================

def check_convergence(
    cost_history: list[float], tolerance: float = 1e-6, window: int = 10
) -> bool:
    """
    Détecter un arrêt empirique par stabilité du coût sur une fenêtre glissante.

    Critère :
        max(J_recent) - min(J_recent) < tolerance

    IMPORTANT : ceci est un CRITÈRE D'ARRÊT EMPIRIQUE, pas une preuve
    mathématique de convergence. Dans le rapport, le nommer explicitement
    "critère d'arrêt par stabilité de la fonction objectif" et non
    "critère de convergence". La convergence théorique du SGD dépend du
    schedule du taux d'apprentissage (voir compute_learning_rate).

    Paramètres
    ----------
    cost_history : list[float]
        Historique des coûts mini-batch.
    tolerance : float
        Seuil d'oscillation accepté.
    window : int > 1
        Nombre d'itérations récentes à considérer.
    """
    if window <= 1:
        raise ValueError("window doit être supérieur à 1.")
    if len(cost_history) < window:
        return False
    recent = cost_history[-window:]
    return (max(recent) - min(recent)) < tolerance


# ============================================================
# 7. CALCUL DU COÛT (INTERNE)
# ============================================================

def _compute_cost(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float,
    lambda_risk: float,
    annualization_factor: int,
    use_cash: bool,
) -> float:
    """Calcul interne du coût, avec ou sans actif sans risque."""
    if use_cash:
        return mean_variance_cost_with_cash(
            weights_risky=weights,
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            risk_free_rate=risk_free_rate,
            lambda_risk=lambda_risk,
            annualization_factor=annualization_factor,
        )
    return mean_variance_cost(
        weights=weights,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        lambda_risk=lambda_risk,
        annualization_factor=annualization_factor,
    )


# ============================================================
# 8. ALGORITHME SGD PRINCIPAL
# ============================================================

def run_sgd(
    returns_matrix: np.ndarray,
    initial_weights: np.ndarray,
    n_iterations: int,
    learning_rate: float,
    lr_schedule: str = "constant",
    gradient_function=None,
    lambda_risk: float = 1.0,
    risk_free_rate: float = 0.0,
    use_cash: bool = True,
    batch_size: int = 64,
    batch_shrinkage: float = 0.1,
    constraint_function=None,
    nonnegative: bool = True,
    exact_simplex: bool = True,
    annualization_factor: int = 252,
    convergence_tolerance: float = 1e-4,
    convergence_window: int = 10,
    eval_every: int = 10,
    random_state: int | None = None,
    verbose: bool = False,
) -> dict:
    """
    Optimiser un portefeuille par SGD avec ré-estimation stochastique des
    paramètres de marché.

    Description de l'algorithme
    ---------------------------
    À chaque itération k :
      1. Tirer un mini-batch de B dates sans remise.
      2. Estimer mu_batch et Sigma_batch sur ce mini-batch.
      3. Calculer le gradient stochastique sur (mu_batch, Sigma_batch).
      4. Mettre à jour : w <- w - eta_k * grad.
      5. Projeter w sur l'ensemble admissible.
      6. Évaluer le coût sur l'ensemble complet tous les eval_every pas.
      7. Tester le critère d'arrêt empirique.

    Nature du SGD
    -------------
    Ce n'est pas un SGD au sens strict (la fonction objectif n'est pas une
    somme de pertes individuelles). Les paramètres mu et Sigma sont ré-estimés
    sur chaque mini-batch, ce qui introduit un bruit d'estimation dans le
    gradient. Cela améliore la robustesse sans garantir la convergence
    théorique sauf si lr_schedule="inverse" (conditions de Robbins-Monro).

    Paramètres principaux
    ---------------------
    returns_matrix : array, forme (T, n)
        Rendements journaliers des n actifs risqués.
    initial_weights : array, forme (n,)
        Poids initiaux sur les actifs risqués.
    n_iterations : int
        Nombre maximal d'itérations.
    learning_rate : float
        Taux de base eta_0.
    lr_schedule : {"constant", "inverse", "sqrt"}
        Schedule du taux d'apprentissage.
        "constant" : eta_k = eta_0 (rapide, peut osciller).
        "inverse"  : eta_k = eta_0/(1+k) (convergence théorique garantie).
        "sqrt"     : eta_k = eta_0/sqrt(1+k) (compromis).
    lambda_risk : float > 0
        Aversion au risque.
    risk_free_rate : float
        Taux sans risque annuel.
    use_cash : bool
        True  => objectif avec actif sans risque, contrainte sum(w) <= 1.
        False => objectif classique, contrainte sum(w) = 1.
    batch_size : int
        Taille du mini-batch B.
    batch_shrinkage : float dans [0,1]
        Régularisation de la covariance estimée sur chaque mini-batch.
        Recommandé > 0 pour les petits batchs (évite une Sigma mal conditionnée).
    gradient_function : callable ou None
        Fonction de gradient personnalisée de signature :
            gradient_function(w, mu_batch, Sigma_batch,
                              risk_free_rate, lambda_risk, annualization_factor)
        Si None, utilise le gradient analytique mean-variance.
    nonnegative : bool
        Si True, interdit les poids négatifs.
    exact_simplex : bool
        Si True et use_cash=False, projection exacte de Duchi sur le simplexe.
    convergence_tolerance : float
        Seuil pour le critère d'arrêt empirique.
    convergence_window : int
        Fenêtre pour le critère d'arrêt empirique.
    eval_every : int
        Fréquence (en itérations) d'évaluation du coût sur données complètes.
    random_state : int ou None
        Graine pour la reproductibilité.
    verbose : bool
        Si True, affiche un log toutes les 50 itérations.

    Retour
    ------
    dict avec :
      - final_weights     : poids finaux sur les actifs risqués (n,)
      - w_rf              : poids final de l'actif sans risque
      - cost_history      : coût mini-batch à chaque itération
      - full_cost_history : coût sur données complètes tous les eval_every pas
      - full_cost_iters   : indices des itérations correspondantes
      - weights_history   : historique des poids risqués
      - w_rf_history      : historique du poids sans risque
      - lr_history        : historique des taux d'apprentissage effectifs
      - n_iterations      : nombre d'itérations effectivement réalisées
      - converged         : True si critère d'arrêt empirique déclenché
      - use_cash          : paramètre utilisé
      - risk_free_rate    : paramètre utilisé
      - lambda_risk       : paramètre utilisé
      - lr_schedule       : schedule utilisé
    """
    # --- Validation des paramètres ---
    if n_iterations <= 0:
        raise ValueError("n_iterations doit être strictement positif.")
    if learning_rate <= 0:
        raise ValueError("learning_rate doit être strictement positif.")
    if lambda_risk <= 0:
        raise ValueError("lambda_risk doit être strictement positif.")
    if batch_size <= 0:
        raise ValueError("batch_size doit être strictement positif.")
    if eval_every <= 0:
        raise ValueError("eval_every doit être strictement positif.")
    if not 0.0 <= batch_shrinkage <= 1.0:
        raise ValueError("batch_shrinkage doit appartenir à [0, 1].")
    if lr_schedule not in ("constant", "inverse", "sqrt"):
        raise ValueError("lr_schedule doit être 'constant', 'inverse' ou 'sqrt'.")

    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    if R.ndim != 2:
        raise ValueError("returns_matrix doit être une matrice (T, n).")
    if not np.isfinite(R).all():
        raise ValueError("returns_matrix contient des NaN ou des valeurs infinies.")

    T_obs, n_assets = R.shape
    if T_obs < 2:
        raise ValueError("Il faut au moins deux observations de rendement.")

    w = np.asarray(initial_weights, dtype=float).copy()
    if w.shape != (n_assets,):
        raise ValueError(
            f"initial_weights doit avoir la forme ({n_assets},), reçu {w.shape}."
        )
    if not np.isfinite(w).all():
        raise ValueError("initial_weights contient des NaN ou des valeurs infinies.")

    # Projection initiale cohérente avec le mode choisi.
    w = apply_constraints(
        w,
        constraint_function=constraint_function,
        nonnegative=nonnegative,
        allow_cash=use_cash,
        exact_simplex=exact_simplex,
    )

    rng = np.random.default_rng(random_state)
    effective_batch_size = min(batch_size, T_obs)

    # Paramètres sur l'ensemble complet (pour évaluation périodique).
    mu_full = R.mean(axis=0)
    Sigma_full = estimate_covariance_matrix(R, shrinkage=0.0)

    # Historiques.
    cost_history: list[float] = []
    full_cost_history: list[float] = []
    full_cost_iters: list[int] = []
    weights_history: list[np.ndarray] = []
    w_rf_history: list[float] = []
    lr_history: list[float] = []

    converged = False
    last_iter = -1

    for k in range(n_iterations):
        last_iter = k

        # Taux d'apprentissage effectif à l'itération k.
        eta_k = compute_learning_rate(learning_rate, k, schedule=lr_schedule)
        lr_history.append(eta_k)

        # 1. Tirage du mini-batch sans remise.
        idx = rng.choice(T_obs, size=effective_batch_size, replace=False)
        batch = R[idx, :]

        # 2. Estimation de mu et Sigma sur le batch.
        mu_batch = batch.mean(axis=0)
        if effective_batch_size < 2:
            Sigma_batch = np.zeros((n_assets, n_assets))
        else:
            Sigma_batch = estimate_covariance_matrix(
                batch, shrinkage=batch_shrinkage
            )

        # 3. Gradient stochastique sur (mu_batch, Sigma_batch).
        if gradient_function is None:
            if use_cash:
                grad = mean_variance_gradient_with_cash(
                    weights_risky=w,
                    expected_returns=mu_batch,
                    covariance_matrix=Sigma_batch,
                    risk_free_rate=risk_free_rate,
                    lambda_risk=lambda_risk,
                    annualization_factor=annualization_factor,
                )
            else:
                grad = mean_variance_gradient(
                    weights=w,
                    expected_returns=mu_batch,
                    covariance_matrix=Sigma_batch,
                    lambda_risk=lambda_risk,
                    annualization_factor=annualization_factor,
                )
        else:
            grad = gradient_function(
                w,
                mu_batch,
                Sigma_batch,
                risk_free_rate,
                lambda_risk,
                annualization_factor,
            )
            grad = np.asarray(grad, dtype=float)
            if grad.shape != w.shape:
                raise ValueError(
                    "gradient_function retourne un gradient de mauvaise forme."
                )

        # 4. Mise à jour avec taux d'apprentissage schedulé.
        w = sgd_update(w, grad, eta_k)

        # 5. Projection sur l'ensemble admissible.
        w = apply_constraints(
            w,
            constraint_function=constraint_function,
            nonnegative=nonnegative,
            allow_cash=use_cash,
            exact_simplex=exact_simplex,
        )

        w_rf = compute_risk_free_weight(w)

        # 6. Historiques.
        cost_batch = _compute_cost(
            weights=w,
            expected_returns=mu_batch,
            covariance_matrix=Sigma_batch,
            risk_free_rate=risk_free_rate,
            lambda_risk=lambda_risk,
            annualization_factor=annualization_factor,
            use_cash=use_cash,
        )
        cost_history.append(cost_batch)
        weights_history.append(w.copy())
        w_rf_history.append(w_rf)

        if k % eval_every == 0:
            cost_full = _compute_cost(
                weights=w,
                expected_returns=mu_full,
                covariance_matrix=Sigma_full,
                risk_free_rate=risk_free_rate,
                lambda_risk=lambda_risk,
                annualization_factor=annualization_factor,
                use_cash=use_cash,
            )
            full_cost_history.append(cost_full)
            full_cost_iters.append(k)

        # 7. Critère d'arrêt empirique (stabilité du coût mini-batch).
        if check_convergence(cost_history, convergence_tolerance, convergence_window):
            converged = True
            if verbose:
                print(
                    f"Arrêt empirique détecté à l'itération {k} "
                    f"(stabilité du coût sur {convergence_window} itérations) : "
                    f"J_batch = {cost_batch:.6f}, eta = {eta_k:.2e}, w_rf = {w_rf:.4f}"
                )
            break

        if verbose and k % 50 == 0:
            print(
                f"Iter {k:>5d} | "
                f"J_batch = {cost_batch:+.6f} | "
                f"||grad|| = {np.linalg.norm(grad):.4e} | "
                f"eta = {eta_k:.2e} | "
                f"sum(w) = {w.sum():.4f} | "
                f"w_rf = {w_rf:.4f}"
            )

    final_w_rf = compute_risk_free_weight(w)

    return {
        "final_weights": w,
        "w_rf": final_w_rf,
        "cost_history": cost_history,
        "full_cost_history": full_cost_history,
        "full_cost_iters": full_cost_iters,
        "weights_history": weights_history,
        "w_rf_history": w_rf_history,
        "lr_history": lr_history,
        "n_iterations": last_iter + 1,
        "converged": converged,
        "use_cash": use_cash,
        "risk_free_rate": risk_free_rate,
        "lambda_risk": lambda_risk,
        "annualization_factor": annualization_factor,
        "lr_schedule": lr_schedule,
    }