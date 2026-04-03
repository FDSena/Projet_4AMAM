"""
crr.py

Description
-----------
Ce module implémente le modèle binomial de Cox-Ross-Rubinstein (CRR)
pour le pricing d’options.

À partir :
- d’un prix initial du sous-jacent
- d’une option définie dans option.py
- des paramètres calibrés dans calibration.py

il permet de :
- construire l’arbre binomial des prix
- calculer les payoffs à maturité
- remonter l’arbre par rétropropagation
- obtenir le prix théorique de l’option

Ce module peut être utilisé pour :
- les options européennes
- éventuellement les options américaines (extension)
"""


# ============================================================
# IMPORTS
# ============================================================

# import numpy as np


# ============================================================
# 1. CONSTRUCTION DE L’ARBRE DES PRIX DU SOUS-JACENT
# ============================================================

def build_price_tree(S0, u, d, n_steps):
    """
    Construire l’arbre binomial des prix du sous-jacent.

    Paramètres
    ----------
    S0 : float
        Prix initial du sous-jacent à la date t = 0.

    u : float
        Facteur de hausse.

    d : float
        Facteur de baisse.

    n_steps : int
        Nombre d’étapes dans l’arbre binomial.

    Retour
    ------
    price_tree : structure de données
        Arbre contenant les prix possibles du sous-jacent
        à chaque date et à chaque nœud.

    Ce qu’il faut faire
    -------------------
    1. Initialiser le prix à la racine avec S0
    2. Pour chaque niveau de l’arbre :
        - calculer les prix après hausse
        - calculer les prix après baisse
    3. Organiser les valeurs dans une structure lisible
       (liste de listes, tableau triangulaire, etc.)

    Formule
    -------
    Au nœud (i, j), le prix peut s’écrire sous la forme :
    S(i, j) = S0 * u^j * d^(i-j)

    où :
    - i est le nombre total d’étapes
    - j est le nombre de hausses
    """
    pass


# ============================================================
# 2. CALCUL DES PAYOFFS A MATURITE
# ============================================================

def compute_terminal_payoffs(price_tree, option):
    """
    Calculer les payoffs de l’option à maturité.

    Paramètres
    ----------
    price_tree : structure de données
        Arbre des prix du sous-jacent.

    option : Option
        Objet option défini dans option.py

    Retour
    ------
    terminal_payoffs : list or array
        Valeurs du payoff à chaque feuille de l’arbre.

    Ce qu’il faut faire
    -------------------
    1. Récupérer les prix du sous-jacent au dernier niveau de l’arbre
    2. Appliquer la fonction payoff de l’option à chacun de ces prix
    3. Retourner la liste des payoffs terminaux

    Remarque
    --------
    Cette étape représente la condition terminale du problème.
    """
    pass


# ============================================================
# 3. RETROPROPAGATION POUR LE PRICING EUROPEEN
# ============================================================

def backward_induction(price_tree, option, r, dt, p_star):
    """
    Effectuer la rétropropagation dans l’arbre pour obtenir
    le prix d’une option européenne.

    Paramètres
    ----------
    price_tree : structure de données
        Arbre des prix du sous-jacent.

    option : Option
        Objet option.

    r : float
        Taux sans risque.

    dt : float
        Pas de temps.

    p_star : float
        Probabilité risque-neutre.

    Retour
    ------
    option_price : float
        Prix théorique de l’option à t = 0.

    value_tree : structure de données
        Arbre des valeurs de l’option à chaque nœud.

    Ce qu’il faut faire
    -------------------
    1. Initialiser les valeurs terminales avec les payoffs
    2. Remonter l’arbre depuis la maturité jusqu’à la racine
    3. À chaque nœud, calculer la valeur actualisée espérée :

       V = exp(-r * dt) * [p_star * V_up + (1 - p_star) * V_down]

    4. Continuer jusqu’au nœud initial
    5. Retourner la valeur en t = 0

    Remarque
    --------
    Pour une option européenne, on ne compare pas avec une valeur d’exercice immédiat.
    """
    pass


# ============================================================
# 4. RETROPROPAGATION POUR LE PRICING AMERICAIN (OPTIONNEL)
# ============================================================

def backward_induction_american(price_tree, option, r, dt, p_star):
    """
    Effectuer la rétropropagation pour une option américaine.

    Paramètres
    ----------
    price_tree : structure de données
        Arbre des prix du sous-jacent.

    option : Option
        Objet option.

    r : float
        Taux sans risque.

    dt : float
        Pas de temps.

    p_star : float
        Probabilité risque-neutre.

    Retour
    ------
    option_price : float
        Prix théorique de l’option américaine à t = 0.

    value_tree : structure de données
        Arbre des valeurs de l’option.

    Ce qu’il faut faire
    -------------------
    1. Initialiser les payoffs terminaux
    2. Remonter l’arbre
    3. À chaque nœud :
        - calculer la valeur de continuation
        - calculer la valeur d’exercice immédiat
        - prendre le maximum des deux

    Formule
    -------
    V = max(exercice immédiat, valeur de continuation)

    Remarque
    --------
    Cette version est une extension du projet.
    """
    pass


# ============================================================
# 5. FONCTION PRINCIPALE DE PRICING
# ============================================================

def crr_price(S0, option, u, d, r, dt, p_star, american=False):
    """
    Calculer le prix d’une option avec le modèle CRR.

    Paramètres
    ----------
    S0 : float
        Prix initial du sous-jacent.

    option : Option
        Objet option.

    u : float
        Facteur de hausse.

    d : float
        Facteur de baisse.

    r : float
        Taux sans risque.

    dt : float
        Pas de temps.

    p_star : float
        Probabilité risque-neutre.

    american : bool, optional
        Si False : pricing européen
        Si True : pricing américain

    Retour
    ------
    option_price : float
        Prix théorique de l’option.

    value_tree : structure de données
        Arbre des valeurs de l’option.

    Ce qu’il faut faire
    -------------------
    1. Déterminer le nombre d’étapes à partir de dt si nécessaire
       ou le faire passer explicitement dans une version alternative
    2. Construire l’arbre des prix
    3. Choisir la bonne méthode de rétropropagation :
        - européenne
        - américaine
    4. Retourner le prix final et éventuellement l’arbre des valeurs

    Objectif
    --------
    Fournir une interface simple vers le modèle CRR.
    """
    pass


# ============================================================
# 6. FONCTION OPTIONNELLE : AFFICHAGE / EXTRACTION DES ARBRES
# ============================================================

def extract_tree_levels(tree):
    """
    Extraire les niveaux d’un arbre pour analyse ou affichage.

    Paramètres
    ----------
    tree : structure de données
        Arbre des prix ou des valeurs.

    Retour
    ------
    levels : list
        Liste des niveaux de l’arbre.

    Utilité
    -------
    Cette fonction est utile dans les notebooks pour :
    - visualiser l’arbre des prix
    - visualiser l’arbre des valeurs de l’option
    - vérifier que l’algorithme fonctionne correctement
    """
    pass


# ============================================================
# 7. NOTES D’UTILISATION
# ============================================================

"""
Utilisation typique
-------------------
1. Créer une option avec option.py
2. Récupérer les paramètres calibrés avec calibration.py
3. Appeler crr_price(...)
4. Analyser le prix obtenu et éventuellement l’arbre dans un notebook

Exemple logique
---------------
- S0 : prix initial du sous-jacent
- option : call ou put
- sigma, r, dt, u, d, p_star : paramètres calibrés
- sortie : prix théorique de l’option

Séparation des rôles
--------------------
- market_data.py : prépare les données de marché
- calibration.py : estime les paramètres du modèle
- option.py : représente l’option financière
- crr.py : calcule le prix de l’option

Remarque importante
-------------------
Ce module constitue le cœur mathématique du projet.
Il doit être :
- clair
- rigoureux
- facile à tester dans les notebooks
"""