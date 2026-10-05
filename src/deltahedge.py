"""
(c) jfyo2 2026. This project is licensed under the MIT License. 

This file defines functions associated with delta hedging. 
"""

import numpy as np 
import math 
import torch 

from scipy.stats import norm

from src.config import * 


def bsm_delta(S_0, K, r, sigma, T):
    """
    # Computes the theoretical delta of a European call under Black-Scholes-Merton given 
    # initial price S_0, strike price K, risk-free rate r, volatility sigma, at time T
    """
    d1 = (np.log(S_0/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def delta_hedge_batch(asset_price_process_array, K, r, sigma, no_of_calls, MAX_TIME, hedging_interval):
    """"
    # Batch-computes delta hedge processes in the time interval [0,MAX_TIME] for a European
    # call with underlying asset price process 'asset_price_process' and strike price K, where the 
    # risk-free rate is r, the (implied) volatility is sigma, and the market maker has sold 
    # the given 'no_of_calls'. The hedge is rebalanced every time period of length 'hedging_inverval'.
    Returns arrays (hedge_times, hedge_process) where hedge_process[i] is the total number of stocks owned at time hedge_times[i].
    """
    # array of times at which to rebalance
    hedge_times = np.arange(0, MAX_TIME, hedging_interval)

    path_length = len(hedge_times)
    no_paths = asset_price_process_array.shape[0]

    hedge_process_array = np.zeros((no_paths, path_length)) # initialize hedging process; this is the number of stocks bought 

    # resample the price array to match the hedging interval 
    hedge_stride = int(round(hedging_interval / (MAX_TIME / path_length)))
    price_array_resampled = asset_price_process_array[:,::hedge_stride]
    

    for idx, t in enumerate(hedge_times):
        # Compute Black-Scholes implied delta at time
        # Vectorized over array because numpy does vectorized maths fast  
        S_t = price_array_resampled[:,idx]
        delta = bsm_delta(S_t, K, r, sigma, MAX_TIME - t)
        hedge_process_array[:,idx] = delta * no_of_calls
        
    return (hedge_times, hedge_process_array)



def delta_hedge_pnl(asset_price_process_array, S_0, C_0, K, r, sigma, no_of_calls, cost_rate, MAX_TIME, hedging_interval):
    # resample the simulation grid onto the hedge grid before anything indexes into it
    n_sim_steps = asset_price_process_array.shape[1] - 1
    sim_dt = MAX_TIME / n_sim_steps
    
    steps_per_hedge = int(round(hedging_interval / sim_dt))
    assert abs(steps_per_hedge * sim_dt - hedging_interval) < 1e-9, "hedge interval must be a multiple of the simulation step"
    
    price_array_resampled = asset_price_process_array[:, ::steps_per_hedge]
    assert np.allclose(price_array_resampled[:, 0], S_0)

    # now we can do the actual delta hedge...
    (hedge_times, hedge_process_array) = delta_hedge_batch(price_array_resampled, K, r, sigma, no_of_calls, MAX_TIME, hedging_interval)
    
    N_PATHS, N_STEPS = hedge_process_array.shape
    assert price_array_resampled.shape[1] == N_STEPS + 1, "need hedge dates 0..T-dt plus the terminal price"

    # Initialize B_0 = C_0 n - N_0 S_0 - c |N_0| S_0
    N_0 = hedge_process_array[:, 0]
    B = C_0 * no_of_calls - N_0 * S_0 - cost_rate * np.abs(N_0) * S_0

    # Update: B_i = B_{i-1} e^{r dt} - (N_i - N_{i-1}) S_i - c |N_i - N_{i-1}| S_i
    for idx in range(1, N_STEPS):
        N_i, N_iminus1 = hedge_process_array[:, idx], hedge_process_array[:, idx - 1]
        S_i = price_array_resampled[:, idx]
        B = B * np.exp(r * hedging_interval) - (N_i - N_iminus1) * S_i - cost_rate * np.abs(N_i - N_iminus1) * S_i
    B = B * np.exp(r * hedging_interval)            # accrue over the final interval

    # Final update 
    S_T = price_array_resampled[:, -1]
    N_final = hedge_process_array[:, -1]
    
    return B + N_final * S_T - cost_rate * np.abs(N_final) * S_T - no_of_calls * np.maximum(S_T - K, 0)



# This is a rewrite of delta_hedge_pnl but with torch operations 
# and adapted to use the path masking that we design in LSTMmodel.simulate_process() (see deephedging.py) 
# Unlike with delta_hedge_pnl, the LSTM model already resamples the simulation grid for us 
# so we do not need to do any resampling. 
def delta_hedge_pnl_torch(asset_price_process_array, hedge_process_array, path_mask_array, C_0_array, K, 
                          r, no_of_calls, cost_rate, hedging_interval):

    N_PATHS, MAX_STEPS = hedge_process_array.shape
    assert C_0_array.shape  == (N_PATHS,)

    N_0 = hedge_process_array[:, 0]
    S_0 = asset_price_process_array[:, 0]
    B = (C_0_array * no_of_calls) - (N_0 * S_0) - (cost_rate * torch.abs(N_0) * S_0)

    for idx in range(1, MAX_STEPS):
        path_mask = path_mask_array[:,idx-1]
        N_i, N_iminus1 = hedge_process_array[:,idx], hedge_process_array[:,idx-1]
        S_i = asset_price_process_array[:,idx]

        exponent = torch.where(path_mask, math.exp(r * hedging_interval), torch.ones_like(N_i))

        N_change = N_i - N_iminus1
        difference = torch.where(path_mask, N_change * S_i + cost_rate * 
                                 torch.abs(N_change) * S_i, torch.zeros_like(N_i))

        B = B * exponent - difference 


    n_steps = path_mask_array.sum(dim=1).long()               
    last = n_steps.clamp(max=MAX_STEPS - 1).unsqueeze(1)      # index at which each path expires  
    
    S_T = asset_price_process_array.gather(1, last).squeeze(1)
    N_final = hedge_process_array.gather(1, last).squeeze(1)
    # note to self. gather(1, last) means: along dimension 1 (the time axis), 
    # for each row i, take the element at column last[i, 0]. 
    # The result has shape (N_PATHS, 1), and .squeeze(1) flattens it to (N_PATHS,).
    # whereas unsqueeze() earlier expands by adding an extra dimension e.g. [1,2,3,4] -> [[1],[2],[3],[4]]

    final_stock_value = N_final * S_T
    liquidation_cost = cost_rate * torch.abs(N_final) * S_T
    payoff = no_of_calls * torch.clamp(S_T - K, min=0)

    return B + final_stock_value - liquidation_cost - payoff 

