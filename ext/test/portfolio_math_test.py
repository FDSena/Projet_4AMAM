from ext.portfolio_math import (
    estimate_expected_returns,
    estimate_covariance_matrix,
    check_weights,
    normalize_weights,
    project_to_nonnegative_weights,
    portfolio_expected_return,
    portfolio_variance,
    portfolio_volatility,
    mean_variance_cost,
    target_return_cost,
    add_risk_free_asset,
    enforce_portfolio_constraints,
)

import numpy as np
import pandas as pd


# ============================================================
# DONNÉES FACTICES
# ============================================================

np.random.seed(42)

dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
assets = ["AAPL", "MSFT", "GOOGL"]

returns_matrix = pd.DataFrame(
    data=np.random.randn(100, 3) * 0.01,
    index=dates,
    columns=assets
)

weights = np.array([0.3, 0.4, 0.3])


# ============================================================
# 1. RENDEMENTS ESPÉRÉS
# ============================================================

mu = estimate_expected_returns(returns_matrix)

print("Expected returns:")
print(mu)
print("Shape mu:", np.array(mu).shape)


# ============================================================
# 2. MATRICE DE COVARIANCE
# ============================================================

cov_matrix = estimate_covariance_matrix(returns_matrix)

print("\nCovariance matrix:")
print(cov_matrix)
print("Shape covariance:", cov_matrix.shape)


# ============================================================
# 3. CHECK WEIGHTS
# ============================================================

print("\nCheck valid weights:", check_weights(weights))
print("Check invalid weights:", check_weights([0.2, 0.2, 0.2]))
print("Check NaN weights:", check_weights([0.5, np.nan, 0.5]))


# ============================================================
# 4. NORMALIZE WEIGHTS
# ============================================================

raw_weights = np.array([2.0, 3.0, 5.0])
normalized = normalize_weights(raw_weights)

print("\nNormalized weights:", normalized)
print("Sum normalized:", normalized.sum())


# ============================================================
# 5. PROJECT NONNEGATIVE
# ============================================================

negative_weights = np.array([0.5, -0.2, 0.7])
projected = project_to_nonnegative_weights(negative_weights)

print("\nProjected weights:", projected)
print("Sum projected:", projected.sum())
print("All positive:", np.all(projected >= 0))


# ============================================================
# 6. PORTFOLIO EXPECTED RETURN
# ============================================================

portfolio_mu = portfolio_expected_return(weights, mu)

print("\nPortfolio expected return:", portfolio_mu)


# ============================================================
# 7. PORTFOLIO VARIANCE
# ============================================================

portfolio_var = portfolio_variance(weights, cov_matrix)

print("\nPortfolio variance:", portfolio_var)


# ============================================================
# 8. PORTFOLIO VOLATILITY
# ============================================================

portfolio_vol = portfolio_volatility(weights, cov_matrix)

print("\nPortfolio volatility:", portfolio_vol)


# ============================================================
# 9. MEAN-VARIANCE COST
# ============================================================

cost = mean_variance_cost(weights, mu, cov_matrix, lambda_risk=1.0)

print("\nMean-variance cost:", cost)


# ============================================================
# 10. TARGET RETURN COST
# ============================================================

target_cost = target_return_cost(
    weights,
    mu,
    cov_matrix,
    target_return=0.001,
    alpha=10
)

print("\nTarget return cost:", target_cost)


# ============================================================
# 11. ADD RISK-FREE ASSET
# ============================================================

risk_free_rate = 0.0001

new_mu, new_cov = add_risk_free_asset(mu, cov_matrix, risk_free_rate)

print("\nNew expected returns with risk-free asset:")
print(new_mu)
print("Shape new mu:", new_mu.shape)

print("\nNew covariance matrix with risk-free asset:")
print(new_cov)
print("Shape new covariance:", new_cov.shape)


# ============================================================
# 12. ENFORCE PORTFOLIO CONSTRAINTS
# ============================================================

bad_weights = np.array([0.5, -0.3, 0.8])
constrained = enforce_portfolio_constraints(bad_weights, nonnegative=True)

print("\nConstrained weights:", constrained)
print("Sum constrained:", constrained.sum())
print("All positive:", np.all(constrained >= 0))


# ============================================================
# 13. TEST ERREUR DIMENSION
# ============================================================

try:
    portfolio_variance([0.5, 0.5], cov_matrix)
except ValueError as e:
    print("\nErreur attendue dimension:", e)
