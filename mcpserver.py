import numpy as np
import pandas as pd
import json
import yfinance as yf
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from scipy.optimize import minimize, linprog
from fastmcp import FastMCP
from typing import Optional

server = FastMCP("quant-portfolio-engine")

# =========================================================================
# DATA INGESTION LAYER
# =========================================================================

@server.tool()
def get_market_data(tickers: list[str], start_date: str, end_date: str) -> str:
    """
    Fetches historical adjusted closing prices for a list of stock tickers and calculates 
    daily returns, annualized mean returns, annualized covariance matrix, and correlation matrix.
    
    For Indian stocks listed on the NSE, append '.NS' to the ticker (e.g., 'RELIANCE.NS').
    For BSE, append '.BO' (e.g., 'RELIANCE.BO').
    For US stocks, use standard tickers (e.g., 'AAPL', 'MSFT').
    
    Args:
        tickers: List of ticker symbols (e.g., ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']).
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        
    Returns:
        JSON string containing mean_returns, cov_matrix, corr_matrix, and daily_returns.
    """
    try:
        # Download data
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return json.dumps({"error": "No data found for the given tickers and date range."})
            
        # Extract Adjusted Close prices
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            prices = data['Close'] # Fallback
            
        # Handle case where only one ticker is downloaded (pandas returns a Series)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
            
        # Drop rows with NaN values (handles newly listed stocks)
        prices = prices.dropna()
        
        if prices.shape[0] < 2:
            return json.dumps({"error": "Not enough historical data points to calculate returns."})
            
        # Calculate Daily Returns
        daily_returns = prices.pct_change().dropna()
        
        # Annualize statistics (252 trading days)
        mean_returns = daily_returns.mean() * 252
        cov_matrix = daily_returns.cov() * 252
        corr_matrix = daily_returns.corr()
        
        # Convert to JSON serializable lists
        response = {
            "tickers": list(prices.columns),
            "mean_returns": mean_returns.tolist(),
            "cov_matrix": cov_matrix.values.tolist(),
            "corr_matrix": corr_matrix.values.tolist(),
            "daily_returns": daily_returns.values.tolist()
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data: {str(e)}"})


# =========================================================================
# HELPER FUNCTIONS: RESEARCH-GRADE MATH
# =========================================================================

def _ledoit_wolf_shrinkage(returns):
    """Ledoit-Wolf (2004) optimal shrinkage estimator for covariance matrix."""
    T, N = returns.shape
    S = np.cov(returns, rowvar=False, ddof=1)
    
    var = np.diag(S)
    std = np.sqrt(var)
    corr = S / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    
    mean_corr = corr[np.triu_indices(N, k=1)].mean()
    
    target = np.outer(std, std) * mean_corr
    np.fill_diagonal(target, var)
    
    Y = returns - returns.mean(axis=0)
    pi_hat = np.sum([(Y[:, i] * Y[:, j]).var(ddof=1) for i in range(N) for j in range(N)])
    rho_hat = np.sum([(Y[:, i]**2).var(ddof=1) for i in range(N)])
    gamma_hat = np.sum((S - target)**2)
    
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = max(0, min(1, kappa / T))
    
    shrunk_cov = delta * target + (1 - delta) * S
    return shrunk_cov, delta

def _compute_mrc(weights, cov_matrix):
    """Compute Marginal Risk Contribution (MRC) and Component Risk Contribution (CRC)."""
    port_vol = np.sqrt(weights @ cov_matrix @ weights.T)
    mrc = (cov_matrix @ weights) / port_vol
    crc = weights * mrc
    return port_vol, mrc, crc

# =========================================================================
# MCP TOOL DEFINITIONS
# =========================================================================

@server.tool()
def estimate_covariance_matrix(returns: list[list[float]], method: str, ewma_decay: float = 0.94) -> str:
    """Estimates Covariance Matrix. Methods: 'sample', 'ledoit_wolf' (shrinkage), 'ewma' (exponentially weighted). For financial data, ledoit_wolf is highly recommended to invert ill-conditioned matrices."""
    R = np.array(returns)
    
    if method == "sample":
        cov = np.cov(R, rowvar=False, ddof=1)
        meta = "Sample Covariance"
    elif method == "ewma":
        T, N = R.shape
        cov = np.zeros((N, N))
        for t in range(T):
            cov += (ewma_decay**(T - t - 1)) * np.outer(R[t], R[t])
        cov = cov * (1 - ewma_decay)
        meta = "EWMA Covariance"
    elif method == "ledoit_wolf":
        cov, delta = _ledoit_wolf_shrinkage(R)
        meta = f"Ledoit-Wolf Shrinkage Covariance (Shrinkage Intensity: {delta:.4f})"
    else:
        return "Error: Invalid method."
        
    return f"COVARIANCE ESTIMATION ({meta}):\n" + np.array2string(cov, precision=4, suppress_small=True)


@server.tool()
def optimize_mean_variance(
    mean_returns: list[float],
    cov_matrix: list[list[float]],
    objective: str,
    target_return: Optional[float] = None,
    risk_free_rate: float = 0.0,
    bounds: Optional[list[list[float]]] = None
) -> str:
    """Classical Markowitz Mean-Variance Optimization. Solves for Max Sharpe, Min Volatility, or Max Return given target volatility. Handles long/short constraints."""
    mu = np.array(mean_returns)
    S = np.array(cov_matrix)
    N = len(mu)
    if bounds is None:
        bounds = [[0, 1]] * N
        
    def portfolio_vol(w): return np.sqrt(w @ S @ w)
    def neg_sharpe(w): return -(w @ mu - risk_free_rate) / portfolio_vol(w)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    if objective == "efficient_return":
        constraints.append({'type': 'eq', 'fun': lambda w: w @ mu - target_return})
        objective_func = portfolio_vol
    elif objective == "max_sharpe":
        objective_func = neg_sharpe
    else: # min_volatility
        objective_func = portfolio_vol
        
    init_guess = np.array([1/N] * N)
    result = minimize(objective_func, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    w_opt = result.x
    port_ret = w_opt @ mu
    port_vol = portfolio_vol(w_opt)
    sharpe = (port_ret - risk_free_rate) / port_vol
    
    return f"""
    MEAN-VARIANCE OPTIMIZATION ({objective.upper()}):
    Optimal Weights: {np.array2string(w_opt, precision=4, suppress_small=True)}
    Expected Return: {port_ret:.4f}
    Expected Volatility: {port_vol:.4f}
    Sharpe Ratio: {sharpe:.4f}
    """

@server.tool()
def optimize_black_litterman(
    market_weights: list[float],
    cov_matrix: list[list[float]],
    risk_free_rate: float,
    P_matrix: list[list[float]],
    Q_vector: list[float],
    risk_aversion: float = 2.5,
    view_confidences: Optional[list[float]] = None
) -> str:
    """Black-Litterman model. Merges market equilibrium implied returns with subjective investor views to generate robust, stable portfolio weights."""
    w_mkt = np.array(market_weights)
    S = np.array(cov_matrix)
    P = np.array(P_matrix)
    Q = np.array(Q_vector)
    if view_confidences is None:
        view_confidences = [0.5] * len(Q)
    confs = view_confidences
    
    N = len(w_mkt)
    Pi = risk_aversion * S @ w_mkt
    Omega = np.diag([conf * np.dot(P[i], S @ P[i].T) for i, conf in enumerate(confs)])
    
    tau = 1.0 / len(w_mkt)
    tau_S = tau * S
    inv_tau_S = np.linalg.inv(tau_S)
    inv_Omega = np.linalg.inv(Omega)
    
    post_mu = np.linalg.inv(inv_tau_S + P.T @ inv_Omega @ P) @ (inv_tau_S @ Pi + P.T @ inv_Omega @ Q)
    
    def neg_sharpe(w): return -(w @ post_mu - risk_free_rate) / np.sqrt(w @ S @ w)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [[0, 1]] * N
    
    result = minimize(neg_sharpe, w_mkt, method='SLSQP', bounds=bounds, constraints=constraints)
    w_opt = result.x
    
    return f"""
    BLACK-LITTERMAN OPTIMIZATION:
    Prior (Equilibrium) Returns: {np.array2string(Pi, precision=4)}
    Posterior Expected Returns: {np.array2string(post_mu, precision=4)}
    
    Optimal Weights: {np.array2string(w_opt, precision=4)}
    Market Weights:  {np.array2string(w_mkt, precision=4)}
    Active Bets:     {np.array2string(w_opt - w_mkt, precision=4)}
    """

@server.tool()
def optimize_hierarchical_risk_parity(
    cov_matrix: list[list[float]],
    correlation_matrix: list[list[float]]
) -> str:
    """Marcos Lopez de Prado's Hierarchical Risk Parity (HRP). Uses hierarchical clustering to allocate capital without inverting the covariance matrix, highly robust to collinearity."""
    S = np.array(cov_matrix)
    corr = np.array(correlation_matrix)
    
    dist = np.sqrt(0.5 * (1 - corr))
    link = sch.linkage(ssd.squareform(dist), 'single')
    sort_order = sch.leaves_list(link)
    S_sorted = S[sort_order][:, sort_order]
    
    w = np.ones(len(S_sorted))
    clusters = [[i] for i in range(len(S_sorted))]
    
    while len(clusters) > 1:
        merged = clusters.pop()
        left = clusters[-1]
        
        def get_cluster_var(cov, indices):
            cov_slice = cov[np.ix_(indices, indices)]
            ivp = 1.0 / np.diag(cov_slice)
            ivp /= ivp.sum()
            return ivp @ cov_slice @ ivp
        
        var_left = get_cluster_var(S_sorted, left)
        var_merged = get_cluster_var(S_sorted, merged)
        
        alpha = 1 - var_left / (var_left + var_merged)
        
        for i in left: w[i] *= alpha
        for i in merged: w[i] *= (1 - alpha)
        
        clusters[-1] = left + merged
        
    w_final = np.zeros(len(w))
    for original_idx, sorted_idx in enumerate(sort_order):
        w_final[sorted_idx] = w[original_idx]
        
    port_vol, mrc, crc = _compute_mrc(w_final, S)
    
    return f"""
    HIERARCHICAL RISK PARITY (HRP):
    Linkage Method: Single Linkage (Quasi-Diagonalization)
    Optimal Weights: {np.array2string(w_final, precision=4)}
    
    RISK ATTRIBUTION:
    Total Volatility: {port_vol:.4f}
    Component Risk Contribution (CRC): {np.array2string(crc, precision=4)}
    """

@server.tool()
def optimize_cvar(
    returns: list[list[float]],
    confidence_level: float = 0.95,
    bounds: Optional[list[list[float]]] = None
) -> str:
    """Conditional Value at Risk (CVaR / Expected Shortfall) Optimization using Rockafellar-Uryasev linear programming formulation. Minimizes tail risk."""
    R = np.array(returns)
    if bounds is None:
        bounds = [[0, 1]] * R.shape[1]
    alpha = 1.0 - confidence_level
    
    T, nAssets = R.shape
    c = np.concatenate([np.zeros(nAssets), [1], (1/(alpha*T)) * np.ones(T)])
    
    A_eq = np.concatenate([np.ones(nAssets), [0], np.zeros(T)])
    b_eq = [1]
    
    A_ub = np.zeros((T, nAssets + 1 + T))
    A_ub[:, :nAssets] = -R
    A_ub[:, nAssets] = -1
    A_ub[:, nAssets+1:] = -np.eye(T)
    b_ub = np.zeros(T)
    
    bounds_full = bounds + [(None, None)] + [(0, None)] * T
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=[A_eq], b_eq=b_eq, bounds=bounds_full, method='highs')
    w_opt = res.x[:nAssets]
    cvar = res.fun
    
    return f"""
    CONDITIONAL VALUE AT RISK (CVaR) OPTIMIZATION:
    Confidence Level: {confidence_level*100:.0f}%
    Optimal Weights: {np.array2string(w_opt, precision=4)}
    Minimized Expected Shortfall (CVaR): {cvar:.4f}
    """

@server.tool()
def optimize_regularized(
    mean_returns: list[float],
    cov_matrix: list[list[float]],
    lambda_l1: float,
    lambda_l2: float,
    risk_free_rate: float = 0.0,
    bounds: Optional[list[list[float]]] = None
) -> str:
    """Regularized Mean-Variance Optimization. Penalizes gross exposure to induce sparsity (L1/Lasso) or shrinkage (L2/Ridge). Prevents extreme corner solutions."""
    mu = np.array(mean_returns)
    S = np.array(cov_matrix)
    N = len(mu)
    if bounds is None:
        bounds = [[-1, 1]] * N
        
    def objective(w):
        vol = np.sqrt(w @ S @ w)
        sharpe = -(w @ mu - risk_free_rate) / vol
        l1_pen = lambda_l1 * np.sum(np.abs(w))
        l2_pen = lambda_l2 * np.sum((w - (1/N))**2)
        return sharpe + l1_pen + l2_pen
        
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    init = np.array([1/N] * N)
    res = minimize(objective, init, method='SLSQP', bounds=bounds, constraints=constraints)
    w_opt = res.x
    
    return f"""
    REGULARIZED PORTFOLIO OPTIMIZATION:
    L1 Penalty (Gross Exp): {lambda_l1}
    L2 Penalty (Shrinkage): {lambda_l2}
    Optimal Weights: {np.array2string(w_opt, precision=4)}
    Gross Exposure: {np.sum(np.abs(w_opt)):.4f}
    Net Exposure: {np.sum(w_opt):.4f}
    """

@server.tool()
def calculate_portfolio_attribution(
    weights: list[float],
    cov_matrix: list[list[float]]
) -> str:
    """Computes Portfolio Risk Attribution: Volatility, Marginal Risk Contribution (MRC), Component Risk Contribution (CRC), and Diversification Ratio."""
    w = np.array(weights)
    S = np.array(cov_matrix)
    
    port_vol, mrc, crc = _compute_mrc(w, S)
    
    weighted_avg_vol = np.sum(w * np.sqrt(np.diag(S)))
    div_ratio = weighted_avg_vol / port_vol
    
    return f"""
    PORTFOLIO RISK ATTRIBUTION:
    Weights: {np.array2string(w, precision=4)}
    Total Volatility: {port_vol:.4f}
    
    Marginal Risk Contribution (MRC): {np.array2string(mrc, precision=4)}
    Component Risk Contribution (CRC): {np.array2string(crc, precision=4)}
    (Sum of CRC = Total Volatility)
    
    Diversification Ratio: {div_ratio:.4f}
    (DR > 1 indicates diversification benefit is being captured).
    """

if __name__ == "__main__":
    server.run(transport="stdio")