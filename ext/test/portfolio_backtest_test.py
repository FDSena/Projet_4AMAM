from ext.portfolio_backtest import (
    compute_portfolio_returns,
    compute_portfolio_value,
    compute_cumulative_return,
    compute_backtest_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown,
    run_static_backtest,
    run_dynamic_backtest,
    equal_weight_strategy,
    compare_strategies,
    run_portfolio_backtest,
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
# 1. TEST PORTFOLIO RETURNS
# ============================================================

portfolio_returns = compute_portfolio_returns(returns_matrix, weights)

print("Portfolio returns:")
print(portfolio_returns.head())
print("Shape:", portfolio_returns.shape)


# ============================================================
# 2. TEST PORTFOLIO VALUE
# ============================================================

portfolio_value = compute_portfolio_value(portfolio_returns, initial_value=1.0)

print("\nPortfolio value:")
print(portfolio_value.head())
print("Initial value:", portfolio_value.iloc[0])
print("Final value:", portfolio_value.iloc[-1])


# ============================================================
# 3. TEST CUMULATIVE RETURN
# ============================================================

cumulative_return = compute_cumulative_return(portfolio_value)

print("\nCumulative return:")
print(cumulative_return)


# ============================================================
# 4. TEST VOLATILITY
# ============================================================

volatility = compute_backtest_volatility(portfolio_returns)

print("\nAnnualized volatility:")
print(volatility)


# ============================================================
# 5. TEST SHARPE RATIO
# ============================================================

sharpe_ratio = compute_sharpe_ratio(portfolio_returns, risk_free_rate=0.0)

print("\nSharpe ratio:")
print(sharpe_ratio)


# ============================================================
# 6. TEST MAX DRAWDOWN
# ============================================================

max_drawdown = compute_max_drawdown(portfolio_value)

print("\nMax drawdown:")
print(max_drawdown)


# ============================================================
# 7. TEST STATIC BACKTEST
# ============================================================

static_results = run_static_backtest(
    returns_matrix=returns_matrix,
    weights=weights,
    initial_value=1.0,
    risk_free_rate=0.0
)

print("\nStatic backtest keys:")
print(static_results.keys())

print("Static cumulative return:", static_results["cumulative_return"])
print("Static volatility:", static_results["volatility"])
print("Static sharpe:", static_results["sharpe_ratio"])
print("Static max drawdown:", static_results["max_drawdown"])


# ============================================================
# 8. TEST EQUAL WEIGHT STRATEGY
# ============================================================

equal_weights = equal_weight_strategy(n_assets=3)

print("\nEqual weights:")
print(equal_weights)
print("Sum equal weights:", equal_weights.sum())


# ============================================================
# 9. TEST DYNAMIC BACKTEST
# ============================================================

# Poids dynamiques factices :
# ici on garde des poids équipondérés sur toutes les dates
weights_over_time = pd.DataFrame(
    data=np.tile(equal_weights, (len(returns_matrix), 1)),
    index=returns_matrix.index,
    columns=returns_matrix.columns
)

dynamic_results = run_dynamic_backtest(
    returns_matrix=returns_matrix,
    weights_over_time=weights_over_time,
    initial_value=1.0,
    risk_free_rate=0.0
)

print("\nDynamic backtest keys:")
print(dynamic_results.keys())

print("Dynamic cumulative return:", dynamic_results["cumulative_return"])
print("Dynamic volatility:", dynamic_results["volatility"])
print("Dynamic sharpe:", dynamic_results["sharpe_ratio"])
print("Dynamic max drawdown:", dynamic_results["max_drawdown"])


# ============================================================
# 10. TEST COMPARAISON STRATEGIES
# ============================================================

strategy_results = {
    "static_custom": static_results,
    "dynamic_equal": dynamic_results
}

comparison = compare_strategies(strategy_results)

print("\nComparison table:")
print(comparison)


# ============================================================
# 11. TEST PIPELINE COMPLET
# ============================================================

backtest_results = run_portfolio_backtest(
    returns_matrix=returns_matrix,
    optimized_weights=weights,
    dynamic_weights=weights_over_time,
    initial_value=1.0,
    risk_free_rate=0.0
)

print("\nBacktest result keys:")
print(backtest_results.keys())

print("\nStrategies tested:")
print(backtest_results["strategy_results"].keys())

print("\nFull comparison:")
print(backtest_results["comparison"])


# ============================================================
# 12. TEST ERREUR DIMENSION
# ============================================================

try:
    compute_portfolio_returns(returns_matrix, weights=[0.5, 0.5])
except ValueError as e:
    print("\nErreur attendue dimension:")
    print(e)