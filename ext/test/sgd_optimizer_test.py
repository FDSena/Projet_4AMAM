from ext.pgd_optimizer import (
    initialize_weights,
    approximate_gradient,
    sgd_update,
    apply_constraints,
    run_sgd,
    store_weights_history,
    store_cost_history,
    check_convergence,
)

from ext.portfolio_math import (
    estimate_expected_returns,
    estimate_covariance_matrix,
    mean_variance_cost,
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

mu = estimate_expected_returns(returns_matrix)
cov_matrix = estimate_covariance_matrix(returns_matrix)


# ============================================================
# 1. TEST INITIALISATION
# ============================================================

w_equal = initialize_weights(n_assets=3, method="equal")
w_random = initialize_weights(n_assets=3, method="random")

print("Equal weights:", w_equal)
print("Sum equal:", w_equal.sum())

print("\nRandom weights:", w_random)
print("Sum random:", w_random.sum())


# ============================================================
# 2. TEST GRADIENT
# ============================================================

gradient = approximate_gradient(
    cost_function=mean_variance_cost,
    weights=w_equal,
    expected_returns=mu,
    covariance_matrix=cov_matrix,
    lambda_risk=1.0
)

print("\nGradient:")
print(gradient)
print("Gradient shape:", gradient.shape)


# ============================================================
# 3. TEST UPDATE SGD
# ============================================================

updated_weights = sgd_update(
    weights=w_equal,
    gradient=gradient,
    learning_rate=0.01
)

print("\nUpdated weights before constraints:")
print(updated_weights)
print("Sum before constraints:", updated_weights.sum())


# ============================================================
# 4. TEST CONTRAINTES
# ============================================================

constrained_weights = apply_constraints(
    weights=updated_weights,
    constraint_function=enforce_portfolio_constraints
)

print("\nConstrained weights:")
print(constrained_weights)
print("Sum constrained:", constrained_weights.sum())
print("All positive:", np.all(constrained_weights >= 0))


# ============================================================
# 5. TEST HISTORIQUES
# ============================================================

weights_history = None
cost_history = None

weights_history = store_weights_history(weights_history, constrained_weights)

cost_value = mean_variance_cost(
    constrained_weights,
    expected_returns=mu,
    covariance_matrix=cov_matrix,
    lambda_risk=1.0
)

cost_history = store_cost_history(cost_history, cost_value)

print("\nWeights history length:", len(weights_history))
print("Cost history:", cost_history)


# ============================================================
# 6. TEST CONVERGENCE
# ============================================================

fake_cost_history = [0.5, 0.3, 0.2, 0.2000000001]

print("\nConvergence test:", check_convergence(fake_cost_history, tolerance=1e-6))


# ============================================================
# 7. TEST RUN SGD COMPLET
# ============================================================

initial_weights = initialize_weights(n_assets=3, method="equal")

results = run_sgd(
    cost_function=mean_variance_cost,
    initial_weights=initial_weights,
    n_iterations=100,
    learning_rate=0.01,
    constraint_function=enforce_portfolio_constraints,
    expected_returns=mu,
    covariance_matrix=cov_matrix,
    lambda_risk=1.0
)

print("\nFinal weights:")
print(results["final_weights"])

print("Sum final weights:", results["final_weights"].sum())
print("All positive:", np.all(results["final_weights"] >= 0))

print("\nCost history length:", len(results["cost_history"]))
print("First cost:", results["cost_history"][0])
print("Last cost:", results["cost_history"][-1])

print("\nWeights history length:", len(results["weights_history"]))