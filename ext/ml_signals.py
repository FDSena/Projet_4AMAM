"""
ml_signals.py

Description
-----------
Ce module est dédié à la construction de signaux d’investissement
ou d’allocations dynamiques à partir de données de marché.

Son rôle est de :
- construire des features à partir des séries financières
- définir une cible ou une logique de décision
- produire un signal d’investissement simple
  ou directement des poids de portefeuille
- préparer une interface avec l’optimisation et le backtesting

Ce module appartient à la partie extension du projet.
Il peut être utilisé par :
- sgd_optimizer.py
- portfolio_backtest.py
- les notebooks d’analyse

Remarque importante
-------------------
Dans une première version, ce module peut rester simple.
Il n’est pas nécessaire d’implémenter un modèle de machine learning
complexe pour qu’il soit pertinent dans le projet.
"""


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# 1. CONSTRUCTION DES FEATURES
# ============================================================

def build_features(price_data, returns_data=None, window=5):
    """
    Construire des variables explicatives (features)
    à partir des données de marché.

    Paramètres
    ----------
    price_data : DataFrame or Series
        Données de prix des actifs.
    returns_data : DataFrame or Series or None
        Rendements des actifs si déjà calculés.
    window : int
        Fenêtre temporelle pour certaines statistiques glissantes.

    Retour
    ------
    features : DataFrame
        Tableau de features utilisables pour un signal
        ou un modèle simple.
    -------------------
    1. Construire les variables pertinentes
    2. Aligner les dates
    3. Gérer les valeurs manquantes générées par les fenêtres glissantes
    4. Retourner un tableau propre
    """
    if returns_data is not None:
        returns_data = returns_data.copy()
    else:
        returns = price_data.pct_change()

    feat = []

    # Rendement cumulé sur la fenêtre
    rend = returns.rolling(window).sum()
    rend.columns = [f"{col}_rend" for col in returns.columns]
    feat.append(rend)

    # Volatilité
    vol = returns.rolling(window).std()
    vol.columns = [f"{col}_vol" for col in returns.columns]
    feat.append(vol)

    # prix moyenne mobile
    mm = price_data.rolling(window).mean()
    price_vs_mm = (price_data - mm) / mm
    price_vs_mm.columns = [f"{col}_price_vs_mm" for col in price_data.columns]
    feat.append(price_vs_mm)

    # Score du rendement
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    score = (returns - roll_mean) / roll_std
    score.columns = [f"{col}_score" for col in returns.columns]
    feat.append(score)

    # Assemblage
    features = pd.concat(feat, axis=1).dropna()

    return features

# ============================================================
# 2. CONSTRUCTION DE LA CIBLE
# ============================================================

def build_target(returns_data, horizon=1, mode="binary"):
    """
    Construire la variable cible à prédire.

    Paramètres
    ----------
    returns_data : DataFrame or Series
        Rendements futurs ou historiques.
    horizon : int
        Horizon de prédiction.
    mode : str
        Type de cible souhaitée.

    Retour
    ------
    target : Series or DataFrame
        Variable cible.

    Modes possibles
    ---------------
    - "binary" :
        prédire si le rendement futur sera positif ou non
    - "continuous" :
        prédire directement un rendement futur
    - "signal" :
        construire un signal discret (acheter / attendre / réduire)

    Ce qu’il faut faire
    -------------------
    1. Décaler les rendements pour construire une cible future
    2. Transformer cette cible selon le mode choisi
    3. Aligner correctement la cible avec les features

    Remarque
    --------
    Pour une première version, une cible binaire est souvent suffisante.
    """
    # Rendement futur
    future_return = returns_data.shift(-horizon)

    # Mode binaire
    if mode == "binary":
        target = (future_return > 0).astype(int)

    # Mode signal
    elif mode == "signal":
        target = np.sign(future_return)
    
    # Mode continu
    elif mode == "continuous":
        target = future_return

    else :
        raise ValueError(f"Mode {mode} unknown. Choose 'binary', 'signal', or 'continuous'.")
    
    return target.dropna()
        

# ============================================================
# 3. SEPARATION TRAIN / TEST
# ============================================================

def split_train_test(features, target, train_ratio=0.8):
    """
    Séparer les données en ensemble d’entraînement et de test.

    Paramètres
    ----------
    features : DataFrame
        Variables explicatives.
    target : Series or DataFrame
        Variable cible.
    train_ratio : float
        Proportion des données utilisées pour l’entraînement.

    Retour
    ------
    X_train, X_test, y_train, y_test : structures de données
        Données séparées pour l’apprentissage et l’évaluation.

    Ce qu’il faut faire
    -------------------
    1. Respecter l’ordre temporel des données
    2. Éviter de mélanger passé et futur
    3. Retourner des sous-ensembles cohérents

    Remarque
    --------
    En finance, il faut garder une logique chronologique stricte.
    """
    common_index = features.index.intersection(target.index)

    # Alignement
    common_index = features.index.intersection(target.index)
    features = features.loc[common_index]
    target = target.loc[common_index]

    #point de séparation
    split = int(len(features) * train_ratio)

    # Séparation
    X_train = features.iloc[:split]
    X_test = features.iloc[split:]
    Y_train = target.iloc[:split]
    Y_test = target.iloc[split:]
    return X_train, X_test, Y_train, Y_test


# ============================================================
# 4. MODELE SIMPLE DE SIGNAL
# ============================================================

def simple_signal_model(features, threshold=0.0):
    """
    Construire un signal simple à partir de features,
    sans utiliser un modèle ML complexe.

    Paramètres
    ----------
    features : DataFrame
        Variables explicatives.
    threshold : float
        Seuil de décision.

    Retour
    ------
    signals : Series
        Signal d’investissement produit à chaque date.

    Exemples de sortie
    ------------------
    - 1 : investir / renforcer
    - 0 : neutre / attendre
    - -1 : réduire / désinvestir

    Ce qu’il faut faire
    -------------------
    1. Choisir une règle simple basée sur une ou plusieurs features
    2. Produire un signal discret
    3. Retourner une série temporelle

    Remarque
    --------
    Cette fonction permet d’avoir une première version fonctionnelle
    même sans modèle de machine learning avancé.
    """
    signal = (features > threshold).astype(int)
    return signal
    


# ============================================================
# 5. PREDICTION D’UN SCORE OU D’UN SIGNAL
# ============================================================

def predict_signal(model, features):
    """
    Produire un signal d’investissement à partir d’un modèle donné.

    Paramètres
    ----------
    model : object or callable
        Modèle ou règle de décision.
    features : DataFrame
        Variables explicatives.

    Retour
    ------
    signals : Series or array-like
        Signal produit par le modèle.

    Ce qu’il faut faire
    -------------------
    1. Appliquer le modèle aux features
    2. Produire une sortie exploitable
    3. Retourner les signaux alignés dans le temps
    """
    pass


# ============================================================
# 6. CONVERSION D’UN SIGNAL EN POIDS
# ============================================================

def signal_to_weights(signals, n_assets=1, normalize=True):
    """
    Transformer un signal en poids de portefeuille.

    Paramètres
    ----------
    signals : Series or array-like
        Signal d’investissement.
    n_assets : int
        Nombre d’actifs.
    normalize : bool
        Indique s’il faut normaliser les poids.

    Retour
    ------
    weights : DataFrame or array-like
        Poids de portefeuille associés aux signaux.

    Exemples d’interprétation
    -------------------------
    - signal positif -> poids plus élevé
    - signal neutre  -> poids réduit
    - signal négatif -> faible exposition

    Ce qu’il faut faire
    -------------------
    1. Définir une règle de passage du signal vers les poids
    2. Gérer le cas multi-actifs si nécessaire
    3. Normaliser les poids si demandé

    Remarque
    --------
    Cette étape fait le lien entre logique prédictive
    et logique portefeuille.
    """
    pass


# ============================================================
# 7. INTERFACE AVEC UNE FONCTION DE PERTE
# ============================================================

def compute_signal_loss(predicted_signal, realized_returns, mode="simple"):
    """
    Calculer une perte ou un score associé à un signal prédit.

    Paramètres
    ----------
    predicted_signal : Series or array-like
        Signal ou allocation prédite.
    realized_returns : Series or array-like
        Rendements effectivement observés.
    mode : str
        Type de fonction de perte.

    Retour
    ------
    loss_value : float
        Valeur de la perte.

    Idées possibles
    ---------------
    - pénaliser les mauvais signaux
    - mesurer la performance associée au signal
    - intégrer un compromis rendement / risque

    Remarque
    --------
    Cette fonction peut servir si vous voulez relier ce module
    à sgd_optimizer.py dans une logique d’apprentissage.
    """
    pass


# ============================================================
# 8. PIPELINE COMPLET DE GENERATION DE SIGNAUX
# ============================================================

def build_ml_signals(price_data, returns_data=None, window=5, model=None):
    """
    Pipeline complet de génération de signaux d’investissement.

    Paramètres
    ----------
    price_data : DataFrame or Series
        Données de prix.
    returns_data : DataFrame or Series or None
        Rendements déjà calculés.
    window : int
        Fenêtre utilisée pour certaines features.
    model : object or None
        Modèle ou règle utilisée pour la prédiction.

    Retour
    ------
    results : dict
        Structure contenant par exemple :
        - features
        - signaux
        - poids associés

    Ce qu’il faut faire
    -------------------
    1. Construire les features
    2. Construire éventuellement une cible
    3. Définir ou appliquer un modèle
    4. Produire des signaux
    5. Convertir les signaux en poids si nécessaire
    6. Retourner les résultats dans un format simple

    Objectif
    --------
    Fournir une interface utilisable par portfolio_backtest.py
    """
    pass


# ============================================================
# 9. NOTES D’UTILISATION
# ============================================================

"""
Utilisation typique
-------------------
1. Construire des features à partir des prix ou rendements
2. Produire un signal d’investissement simple
3. Convertir ce signal en poids de portefeuille
4. Évaluer la stratégie avec portfolio_backtest.py

Exemple logique
---------------
prix / rendements -> features -> signal -> poids -> backtest

Remarque importante
-------------------
Dans une première version du projet, ce module peut rester très simple.
Il n’est pas nécessaire d’utiliser un vrai modèle supervisé complexe
si une règle de signal simple suffit à illustrer l’idée.

Séparation des rôles
--------------------
- portfolio_math.py : définit les métriques et fonctions de coût
- sgd_optimizer.py : optimise des poids si nécessaire
- ml_signals.py : produit des signaux ou allocations dynamiques
- portfolio_backtest.py : compare les performances des stratégies
"""