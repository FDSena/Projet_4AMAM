"""
option.py

Description
-----------
Ce module définit la structure d’une option financière.

Une option est un produit dérivé qui donne le droit (mais non l’obligation) :
- d’acheter un actif (call)
- de vendre un actif (put)

à un prix fixé à l’avance (strike), à une date donnée (maturité).

Ce module permet de :
- représenter une option sous forme d’objet
- stocker ses caractéristiques principales
- calculer son payoff (utile pour le modèle CRR)

Ce module est utilisé par :
- crr.py (pour le pricing)
"""


# ============================================================
# CLASSE OPTION
# ============================================================

class Option:
    """
    Classe représentant une option financière.
    """

    def __init__(self, strike, maturity, option_type):
        """
        Initialiser une option.

        Paramètres
        ----------
        strike : float
            Prix d’exercice de l’option (K)

        maturity : float
            Maturité de l’option (en années)

        option_type : str
            Type d’option :
            - 'call' pour une option d’achat
            - 'put' pour une option de vente

        Ce qu’il faut faire
        -------------------
        1. Stocker les paramètres dans l’objet
        2. Vérifier que option_type est valide ('call' ou 'put')
        """
        pass


    # ========================================================
    # PAYOFF
    # ========================================================

    def payoff(self, S):
        """
        Calculer le payoff de l’option à maturité.

        Paramètres
        ----------
        S : float
            Prix du sous-jacent à maturité

        Retour
        ------
        payoff : float

        Formules
        --------
        Call :
            max(S - K, 0)

        Put :
            max(K - S, 0)

        Utilité
        -------
        Cette fonction sera utilisée dans crr.py
        pour calculer la valeur terminale de l’option
        sur les feuilles de l’arbre binomial.
        """
        pass


    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self):
        """
        Représentation de l’objet option.

        Objectif
        --------
        Fournir une description lisible de l’option,
        utile pour le debug et les notebooks.

        Exemple attendu
        ----------------
        Option(type='call', strike=100, maturity=1.0)
        """
        pass


# ============================================================
# NOTES IMPORTANTES
# ============================================================

"""
Pourquoi utiliser une classe ici ?
----------------------------------
Une option est un objet naturel en finance, avec :
- des attributs (strike, maturité, type)
- un comportement (payoff)

Cela permet de rendre le code :
- plus lisible
- plus structuré
- plus proche du modèle mathématique

Utilisation typique dans le projet
----------------------------------
1. Créer une option :
    option = Option(strike=100, maturity=1.0, option_type='call')

2. Passer l’option au modèle CRR :
    price = crr_price(option, ...)

3. Calculer le payoff dans l’arbre :
    option.payoff(S)

Remarque
--------
Cette classe reste volontairement simple :
elle ne fait que représenter l’option,
tout le pricing est géré dans crr.py.
"""