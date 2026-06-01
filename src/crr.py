import numpy as np


def build_price_tree(S0, u, d, n_steps):
    tree = []
    for i in range(n_steps + 1):
        level = [S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
        tree.append(level)
    return tree


def compute_terminal_payoffs(price_tree, option):
    terminal_prices = price_tree[-1]
    return [option.payoff(S) for S in terminal_prices]


# 3. RETROPROPAGATION EUROPEENNE

def backward_induction(price_tree, option, r, dt, p_star):
    n_steps = len(price_tree) - 1
    discount = np.exp(-r * dt)

    value_tree = [None] * (n_steps + 1)
    value_tree[n_steps] = compute_terminal_payoffs(price_tree, option)

    for i in range(n_steps - 1, -1, -1):
        level = []
        for j in range(i + 1):
            v = discount * (p_star * value_tree[i + 1][j + 1] + (1 - p_star) * value_tree[i + 1][j])
            level.append(v)
        value_tree[i] = level

    return value_tree[0][0], value_tree


# 4. RETROPROPAGATION AMERICAINE

def backward_induction_american(price_tree, option, r, dt, p_star):
    n_steps = len(price_tree) - 1
    discount = np.exp(-r * dt)

    value_tree = [None] * (n_steps + 1)
    value_tree[n_steps] = compute_terminal_payoffs(price_tree, option)

    for i in range(n_steps - 1, -1, -1):
        level = []
        for j in range(i + 1):
            continuation = discount * (p_star * value_tree[i + 1][j + 1] + (1 - p_star) * value_tree[i + 1][j])
            exercise = option.payoff(price_tree[i][j])
            level.append(max(continuation, exercise))
        value_tree[i] = level

    return value_tree[0][0], value_tree


# 5. FONCTION PRINCIPALE DE PRICING

def crr_price(S0, option, u, d, r, dt, p_star, american=False):
    if S0 <= 0:
        raise ValueError("Le prix initial S0 doit être positif.")
    if dt <= 0:
        raise ValueError("Le pas de temps dt doit être positif.")
    if u <= d:
        raise ValueError("On doit avoir u > d dans le modèle CRR.")
    if not (0 <= p_star <= 1):
        raise ValueError("La probabilité risque-neutre p_star doit être entre 0 et 1.")

    n_steps = round(option.maturity / dt)
    price_tree = build_price_tree(S0, u, d, n_steps)

    if american:
        return backward_induction_american(price_tree, option, r, dt, p_star)
    else:
        return backward_induction(price_tree, option, r, dt, p_star)

# 6. EXTRACTION DES NIVEAUX D'UN ARBRE

def extract_tree_levels(tree):
    return [level for level in tree]