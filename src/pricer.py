import numpy as np 
from scipy.stats import norm


def option_price_mc(times, paths, T, K, r):
    """
    # Returns the Monte Carlo theoretical price of a European call option 
    # at time T with strike price K and risk-free rate r,
    # given Monte Carlo underlying asset price paths 'paths' 
    # indexed at times 'times'
    """

    idx = np.searchsorted(times, T)

    # We will estimate the price at time T by linearly interpolating between the prices 
    # of the asset at t_{n-1} and t_n such that t_{n-1} <= T <= t_n 
    # the value t_n can be found using numpy's built-in np.searchsort
    # For the boundary we need a special case where we don't interpolate because there 
    # is nothing after to interpolate with  
    if times[idx] == T:
        S_T = paths[:, idx] # note S_T is vectorized here
    else:
        t0, t1 = times[idx-1], times[idx]
        S_T = paths[:, idx-1] + (paths[:, idx] - paths[:, idx-1]) / (t1 - t0) * (T - t0)

    payoffs = np.maximum(S_T - K, 0)

    option_price = np.exp(-r * T) * np.mean(payoffs)

    return option_price



def option_price_bsm(S_0, K, T, r, sigma):
    """
    # Returns the theoretical price of a European call option under Black-Scholes-Merton
    # with initial price S_0, strike K, risk-free rate r and volatility sigma 
    # at time T
    """
    d1 = (np.log(S_0/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S_0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
