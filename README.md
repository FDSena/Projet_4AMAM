# Projet_4AMAM

## Estimation de la valeur d’options – Modèle de Cox-Ross-Rubinstein

---

## Description du projet

Ce projet s’inscrit dans le cadre du module de **mathématiques financières** et porte sur l’estimation de la valeur d’options à l’aide du modèle binomial de **Cox–Ross–Rubinstein (CRR)**.

Une option financière donne le droit (sans obligation) d’acheter ou de vendre un actif à un prix fixé à l’avance (strike). Le modèle CRR permet de modéliser l’évolution du prix de l’actif sous forme d’un **arbre binomial discret**, et d’en déduire le prix de l’option.

L’objectif du projet est de :

- Comprendre et implémenter le modèle CRR  
- Calibrer les paramètres du modèle à partir de données réelles  
- Évaluer la pertinence du modèle sur des données de marché  

---

## Objectifs

### Partie 1 — Cœur du projet
- Implémentation du modèle binomial CRR  
- Construction de l’arbre des prix  
- Calcul du payoff des options (call / put)  
- Rétropropagation pour obtenir le prix de l’option  
- Calibration des paramètres (volatilité, taux sans risque)  
- Application sur données réelles (CAC40, Brent, etc.)  
- Évaluation du modèle (erreurs, stabilité)

### Partie 2 — Extension (optionnelle)
- Optimisation de portefeuille multi-actifs  
- Intégration d’un actif sans risque  
- Approche mean–variance  
- Implémentation d’un algorithme de type SGD  
- Utilisation éventuelle de signaux (sentiment, ML)

---

## Structure du projet

## A modifier selon les besoins, mais voici une proposition de structure pour organiser le code et les analyses.

### Cœur du modèle CRR

```text
project_4AMAM/
├── src/
│	├── market_data.py      # Gestion des données de marché (Yahoo Finance, nettoyage, rendements)
│	├── option.py           # Définition des options (call / put, strike, maturité)
│	├── calibration.py      # Estimation des paramètres (volatilité, taux, u, d, p*)
│	├── crr.py              # Implémentation du modèle CRR (arbre + pricing)
│	└── backtester.py       # Évaluation du modèle (MAE, RMSE, stabilité)
```

### Extension
```text
project_4AMAM/
├── Ext/
│	├── sentiment.py          # Analyse de sentiment (news, texte)
│	├── portfolio_math.py     # Modélisation mathématique du portefeuille
│	├── sgd_optimizer.py      # Optimisation par descente de gradient stochastique
│	├── ml_signals.py         # Modèle ML pour signaux ou allocation
│	└── portfolio_backtest.py # Backtesting des stratégies de portefeuille
```

### Analyse des données
```text
Projet_4AMAM/
│
├── notebooks/          # Analyses, visualisations et expérimentations
│   ├── data_analysis.ipynb      # Exploration des données (prix, rendements)
│   ├── calibration.ipynb        # Étude de la volatilité et paramètres CRR
│   ├── pricing_analysis.ipynb   # Analyse du pricing CRR
│   └── backtesting.ipynb        # Analyse des performances du modèle
│   └── portfolio_analysis.ipynb  # Analyse des stratégies de portefeuille
```

### Rapport
```text
project_4AMAM/
├── report/
│   ├── img/				# Graphiques et figures pour le rapport
│   ├── report.tex         # Rapport final du projet
│   └── presentation.pptx  # Présentation pour la soutenance

```

## Références

- Cox, Ross, Rubinstein (1979) — *Option Pricing: A Simplified Approach*  
- Hull, J. C. — *Options, Futures, and Other Derivatives*  
- Investopedia — Binomial Option Pricing Model  
- Supports universitaires (Stanford, MIT, UW)

---

## Lancement du projet

1. Installer les dépendances :
pip install numpy pandas yfinance matplotlib

2. Lancer le script principal :

python main.py

---

## Auteurs

Projet réalisé dans le cadre du cursus **4A MAM – Polytech Lyon**

- Dridi Mohamed Dhia
- Roussel Noah
- DANTAS DE SENA Flavio  

---

## Remarques

Ce projet vise à relier :
- modélisation mathématique  
- implémentation algorithmique  
- données financières réelles  

L’extension proposée permet d’ouvrir vers des problématiques modernes :
- optimisation de portefeuille  
- machine learning  
- finance quantitative avancée  

---

## Perspectives

Le projet met en évidence les limites du modèle CRR (hypothèses dans un marché stable) et ouvre la voie à des modèles plus avancés comme Black-Scholes ou les modèles à volatilité stochastique.