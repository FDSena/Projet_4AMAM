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
 
    price_data = price_data.copy()
 
    if returns_data is not None:
        returns = returns_data.copy()
    else:
        returns = price_data.pct_change()
 
    feat = []
 
    # Rendement cumulé
    rend = returns.rolling(window).sum()
    rend.columns = [f"{col}_rend_{window}" for col in returns.columns]
    feat.append(rend)
 
    # Volatilité glissante
    vol = returns.rolling(window).std()
    vol.columns = [f"{col}_vol_{window}" for col in returns.columns]
    feat.append(vol)
 
    # Momentum
    momentum = price_data / price_data.shift(window) - 1
    momentum.columns = [f"{col}_momentum_{window}" for col in price_data.columns]
    feat.append(momentum)
 
    # prix moyenne mobile
    moving_avg = price_data.rolling(window).mean()
    price_vs_ma = (price_data - moving_avg) / moving_avg
    price_vs_ma.columns = [f"{col}_price_vs_moving_avg_{window}" for col in price_data.columns]
    feat.append(price_vs_ma)
 
    # Score du rendement
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    score = (returns - roll_mean) / roll_std
    score.columns = [f"{col}_score_{window}" for col in returns.columns]
    feat.append(score)
 
    # Assemblage
    features = pd.concat(feat, axis=1)
 
    # Nettoyage
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
 
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
# 3. MODELE SIMPLE DE SIGNAL
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
    mom_cols = [col for col in features.columns if "momentum" in col]
    momentum = features[mom_cols]
 
    if len(momentum.columns) == 0:
        raise ValueError("No momentum features found. Please check the feature names.")
 
    momentum = features[mom_cols]
 
    # moyenne du momentum
    mm = momentum.mean(axis=1)
 
    # règle de décision
    signals = pd.Series(0, index=features.index)
    signals[mm > threshold] = 1
    signals[mm < -threshold] = -1
 
    return signals
    
 
 
# ============================================================
# 5. PREDICTION D’UN SCORE OU D’UN SIGNAL
# ============================================================
 
def predict_signal(model, features, **kwargs):
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
    if not callable(model):
        raise ValueError("Model must be a callable object or function.")
    
    signals = model(features, **kwargs)
 
    if not isinstance(signals,(pd.Series, np.ndarray)):
        raise ValueError("Model output must be a pandas Series or numpy array.")
    
    if len(signals) != len(features):
        raise ValueError("Model output length must match the number of feature rows.")
    
    if isinstance(signals, np.ndarray):
        signals = pd.Series(signals, index=features.index)
    
    return signals
 
 
# ============================================================
# 6. CONVERSION D’UN SIGNAL EN POIDS
# ============================================================
 
def signal_to_weights(signals, n_assets=1, normalize=True, asset_names=None):
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
    asset_names : list or None
        Noms des colonnes à utiliser pour le DataFrame de poids.
        Doit correspondre aux colonnes de returns_matrix dans portfolio_backtest.py.
        Si None, les colonnes seront nommées "Asset_1", "Asset_2", etc.
        IMPORTANT : toujours passer les vrais tickers (ex: ["TTE.PA", "BNP.PA"])
        pour garantir l’alignement avec returns_matrix dans run_dynamic_backtest.
 
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
    if not isinstance(signals, pd.Series):
        signals = pd.Series(signals)
 
    if n_assets == 1:
        weights = signals.copy()
        weights = weights.replace({
            1: 1.0,
            0: 0.0,
            -1: 0.0
        })
        return weights
 
    # Résoudre les noms de colonnes
    if asset_names is not None:
        if len(asset_names) != n_assets:
            raise ValueError(
                f"asset_names a {len(asset_names)} éléments mais n_assets={n_assets}. "
                "Ils doivent correspondre."
            )
        columns = list(asset_names)
    else:
        columns = [f"Asset_{i+1}" for i in range(n_assets)]
 
    weights = pd.DataFrame(
        0.0,
        index=signals.index,
        columns=columns
    )
 
    for date, signal in signals.items():
        if signal == 1:
            weights.loc[date] = 1.0 / n_assets
        elif signal == 0:
            weights.loc[date] = 1.0 / n_assets
        elif signal == -1:
            weights.loc[date] = 0.0
 
    if normalize:
        row_sums = weights.sum(axis=1)
        weights = weights.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
 
    return weights
 
# ============================================================
# 6. PIPELINE COMPLET DE GENERATION DE SIGNAUX
# ============================================================
 
def build_ml_signals(price_data, returns_data=None, window=5, model=None, threshold=0.0, asset_names=None):
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
    asset_names : list or None
        Noms des colonnes des actifs (tickers CAC 40 par exemple).
        Si fourni, les poids générés auront ces noms en colonnes.
        Si None, déduit automatiquement depuis price_data.columns.
        Doit correspondre aux colonnes de returns_matrix utilisé
        dans portfolio_backtest.run_dynamic_backtest pour éviter
        le bug d’alignement (NaN silencieux).
 
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
    features = build_features(price_data, returns_data, window)
 
    if model is None:
        model = simple_signal_model
 
    # prédiction du signal
    signals = predict_signal(
        model=model,
        features=features,
        threshold=threshold
    )
 
    # Determiner le nombre d'actifs
    if isinstance(price_data, pd.DataFrame):
        n_assets = price_data.shape[1]
    else:
        n_assets = 1
 
    # conversion en poids
    # asset_names doit correspondre aux colonnes de returns_matrix
    # pour éviter le bug d'alignement dans run_dynamic_backtest
    if asset_names is None and isinstance(price_data, pd.DataFrame):
        asset_names = list(price_data.columns)
    weights = signal_to_weights(
        signals,
        n_assets=n_assets,
        asset_names=asset_names
    )
 
    results = {
        "features": features,
        "signals": signals,
        "weights": weights
    }
 
    return results
 
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