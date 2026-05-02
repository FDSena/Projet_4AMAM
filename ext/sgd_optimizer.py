"""
sgd_optimizer.py

Description
-----------
Ce module implémente un algorithme d’optimisation de type
descente de gradient stochastique (SGD) pour ajuster les poids
d’un portefeuille.

Son rôle est de :
- initialiser un vecteur de poids
- calculer une mise à jour itérative des poids
- minimiser une fonction de coût définie dans portfolio_math.py
- gérer les contraintes sur les poids
- sauvegarder l’historique de l’optimisation

Ce module appartient à la partie extension du projet.
Il sera utilisé par :
- portfolio_math.py
- portfolio_backtest.py
- les notebooks d’analyse
"""


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
from portfolio_math import check_weights

# ============================================================
# 1. INITIALISATION DES POIDS
# ============================================================

def initialize_weights(n_assets, method="equal"):
    """
    Initialiser les poids du portefeuille.

    Paramètres
    ----------
    n_assets : int
        Nombre total d’actifs dans le portefeuille.

    method : str
        Méthode d’initialisation.
        Exemples :
        - "equal" : poids égaux
        - "random" : poids aléatoires puis normalisés

    Retour
    ------
    weights : array-like
        Vecteur initial de poids.

    Ce qu’il faut faire
    -------------------
    1. Créer un vecteur de taille n_assets
    2. Initialiser les poids selon la méthode choisie
    3. Normaliser les poids pour que leur somme fasse 1

    Remarque
    --------
    Une initialisation simple par poids égaux est souvent suffisante
    pour une première version.
    """
    if n_assets <= 0:
        raise ValueError("n_assets must be a positive integer.")
    
    if method == "equal":
        weights = np.ones(n_assets) / n_assets

    elif method == "random":
        weights = np.random.rand(n_assets)
        weights /= np.sum(weights)

    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    return weights

# ============================================================
# 2. CALCUL APPROCHE DU GRADIENT
# ============================================================

def approximate_gradient(cost_function, weights, epsilon=1e-6, **kwargs):
    """
    Approcher numériquement le gradient de la fonction de coût.

    Paramètres
    ----------
    cost_function : callable
        Fonction de coût à minimiser.
    weights : array-like
        Vecteur courant des poids.
    epsilon : float
        Petite perturbation utilisée pour l’approximation numérique.
    **kwargs :
        Paramètres supplémentaires transmis à la fonction de coût.

    Retour
    ------
    gradient : array-like
        Approximation du gradient de la fonction de coût.

    Ce qu’il faut faire
    -------------------
    1. Perturber chaque composante du vecteur de poids
    2. Évaluer la variation de la fonction de coût
    3. Estimer chaque dérivée partielle
    4. Construire le vecteur gradient

    Remarque
    --------
    Cette approche est simple à comprendre et à implémenter,
    même si elle est moins efficace qu’un gradient analytique.
    Pour un projet étudiant, elle est souvent suffisante.
    """
    weights = np.array(weights, dtype=float)
    gradient = np.zeros_like(weights)

    base_cost = cost_function(weights, **kwargs)
    n = len(weights)
    for i in range(n):
        perturbed_weights = weights.copy()
        perturbed_weights[i] += epsilon

        perturbed_cost = cost_function(perturbed_weights, **kwargs)
        gradient[i] = (perturbed_cost - base_cost) / epsilon

    return gradient


# ============================================================
# 3. MISE A JOUR DES POIDS
# ============================================================

def sgd_update(weights, gradient, learning_rate):
    """
    Effectuer une mise à jour des poids selon la règle du SGD.

    Paramètres
    ----------
    weights : array-like
        Poids courants.
    gradient : array-like
        Gradient de la fonction de coût.
    learning_rate : float
        Pas d’apprentissage.

    Retour
    ------
    new_weights : array-like
        Poids mis à jour.

    Formule
    -------
    w_new = w - eta * gradient

    où eta est le learning rate.

    Utilité
    -------
    Cette fonction réalise le cœur de la descente de gradient.
    """
    weights = np.array(weights, dtype=float)
    gradient = np.array(gradient, dtype=float)

    if weights.shape != gradient.shape:
        raise ValueError("Weights and gradient must have the same shape.")
    
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")
    
    new_weights = weights - learning_rate * gradient

    return new_weights

# ============================================================
# 4. APPLICATION DES CONTRAINTES
# ============================================================

def apply_constraints(weights, constraint_function=None):
    """
    Appliquer les contraintes après une mise à jour des poids.

    Paramètres
    ----------
    weights : array-like
        Poids mis à jour.
    constraint_function : callable or None
        Fonction imposant les contraintes du portefeuille.

    Retour
    ------
    constrained_weights : array-like
        Poids corrigés après application des contraintes.

    Ce qu’il faut faire
    -------------------
    1. Si une fonction de contrainte est fournie, l’appliquer
    2. Sinon, au minimum renormaliser les poids
    3. Retourner les poids valides

    Utilité
    -------
    Assure que les poids restent interprétables
    et respectent les règles fixées dans le projet.
    """
    weights = np.array(weights, dtype=float)
    if constraint_function is not None:
        if not callable(constraint_function):
            raise ValueError("constraint_function must be callable.")
        
        constrained_weights = constraint_function(weights)

    else:
        sum_weights = np.sum(weights)

        if not np.isclose(sum_weights, 0.0):
            constrained_weights = weights / sum_weights
        
        else:
            n=len(weights)
            constrained_weights = np.ones(n) / n
    return constrained_weights

# ============================================================
# 5. SGD PRINCIPAL
# ============================================================

def run_sgd(
    cost_function,
    initial_weights,
    n_iterations,
    learning_rate,
    constraint_function=None,
    gradient_function=None,
    **kwargs
):
    """
    Exécuter l’algorithme de descente de gradient stochastique.

    Paramètres
    ----------
    cost_function : callable
        Fonction de coût à minimiser.

    initial_weights : array-like
        Poids de départ.

    n_iterations : int
        Nombre d’itérations de l’algorithme.

    learning_rate : float
        Pas d’apprentissage.

    constraint_function : callable or None
        Fonction utilisée pour imposer les contraintes sur les poids.

    gradient_function : callable or None
        Fonction de calcul du gradient.
        Si None, utiliser une approximation numérique.

    **kwargs :
        Paramètres supplémentaires transmis à la fonction de coût
        et éventuellement à la fonction de gradient.

    Retour
    ------
    results : dict
        Dictionnaire contenant par exemple :
        - final_weights
        - cost_history
        - weights_history

    Ce qu’il faut faire
    -------------------
    1. Initialiser les poids
    2. Répéter sur n_iterations :
        - calculer le gradient
        - mettre à jour les poids
        - appliquer les contraintes
        - calculer la nouvelle valeur de la fonction de coût
        - sauvegarder l’historique
    3. Retourner les résultats de l’optimisation

    Objectif
    --------
    Fournir une interface générale d’optimisation utilisable
    avec plusieurs fonctions de coût.
    """
    if not callable(cost_function):
        raise ValueError("cost_function must be callable.")
    
    if n_iterations <= 0:
        raise ValueError("n_iterations must be a positive integer.")
    
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    
    weights = np.array(initial_weights, dtype=float)

    weights_history = []
    cost_history = []
    
    for _ in range(n_iterations):
        if gradient_function is None:
            gradient = approximate_gradient(
                cost_function,
                weights,
                **kwargs
            )
        else:
            gradient = gradient_function(weights, **kwargs)

        weights = sgd_update(
            weights,
            gradient,
            learning_rate
        )

        weights = apply_constraints(
            weights,
            constraint_function
        )

        if not check_weights(weights):
            raise ValueError(f"Poids invalides après apply_constraints : {weights}")

        cost_value = cost_function(
            weights,
            **kwargs
        )

        weights_history.append(weights.copy())
        cost_history.append(cost_value)
        
        if check_convergence(cost_history):
            break



    results = {
        "final_weights": weights,
        "cost_history": cost_history,
        "weights_history": weights_history
    }

    return results
    

# ============================================================
# 8. CRITERE D’ARRET OPTIONNEL
# ============================================================

def check_convergence(cost_history, tolerance=1e-6):
    """
    Vérifier si l’algorithme a convergé.

    Paramètres
    ----------
    cost_history : list
        Historique des valeurs de la fonction de coût.
    tolerance : float
        Seuil de variation minimale.

    Retour
    ------
    converged : bool
        Indique si l’optimisation peut être arrêtée.

    Ce qu’il faut faire
    -------------------
    1. Comparer les dernières valeurs de la fonction de coût
    2. Déterminer si la variation est suffisamment petite

    Remarque
    --------
    Cette fonction est optionnelle.
    Dans une première version, on peut se contenter
    d’un nombre fixe d’itérations.
    """
    if len(cost_history) < 2:
        return False

    last_cost = cost_history[-1]
    prev_cost = cost_history[-2]

    var = abs(last_cost - prev_cost)
    return var < tolerance    


# ============================================================
# 9. NOTES D’UTILISATION
# ============================================================

"""
Utilisation typique
-------------------
1. Définir une fonction de coût dans portfolio_math.py
2. Initialiser les poids du portefeuille
3. Lancer run_sgd(...)
4. Récupérer :
    - les poids optimisés
    - l’historique du coût
    - l’historique des poids

Exemple logique
---------------
cost function + initial weights -> gradient -> update -> constraints -> repeat

Remarque importante
-------------------
Ce module réalise l’optimisation,
mais il ne définit pas lui-même la fonction de coût.
La logique mathématique reste dans portfolio_math.py.

Séparation des rôles
--------------------
- portfolio_math.py : définit les fonctions de coût et les contraintes
- sgd_optimizer.py : optimise les poids
- portfolio_backtest.py : évalue les portefeuilles obtenus
"""