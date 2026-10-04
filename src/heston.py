"""
(c) jfyo2 2026. This project is licensed under the MIT License. 

This file defines functions associated with Heston path simulation. 
See: https://en.wikipedia.org/wiki/Heston_model
"""


import numpy as np 


def correlatedBM(RNG, rho, MAX_TIME=1, SAMPLES=100):
    """
    # Creates two correlated Brownian motions with correlation 'rho'
    # in the time interval [0, MAX_TIME].
    # Output: (t, W1, W2) where t is an array of time values 
    # and W1, W2 are arrays containing the values of the Brownian motions at the 
    # points in t
    """
    # To do this we can use the fact that if W1 and Z are independent standard Brownian motions then 
    # W2 = \rho W1 + \sqrt{1 - \rho^2} Z is a Brownian motion with correlation rho with W1
    # See: https://quant.stackexchange.com/questions/24472/two-correlated-brownian-motions

    t = np.linspace(0, MAX_TIME, SAMPLES)
    dt = MAX_TIME / (SAMPLES - 1) 

    # Generate two independent standard BMs
    # First generate normal samples
    dW1 = RNG.normal(0, np.sqrt(dt), SAMPLES - 1)
    dZ  = RNG.normal(0, np.sqrt(dt), SAMPLES - 1)
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dZ

    # Make the initial values 0; this is a standard BM 
    W1 = np.concatenate(([0.0], np.cumsum(dW1)))
    W2 = np.concatenate(([0.0], np.cumsum(dW2)))

    return (t, W1, W2)

    
def hestonSim(RNG, S_0, nu_0, mu, kappa, theta, xi, rho, MAX_TIME=1, SAMPLES=100):
    """
    # Simulates Heston model of asset price using Euler-Maruyama. 
    # Output: (t, S, nu) where t is a vector of time intervals at which we sample, S 
    # is the discretized price process, and nu is the discretized volatility process.
    """
    # Feller condition warning. Code is set up so that it will clip values to 0 if Feller 
    # is violated, so we only warn and do not raise an error
    if 2 * kappa * theta <= xi**2: 
        print("Warning! These values of kappa, theta, and xi violate the Feller condition.")

    (t, W_S, W_nu) = correlatedBM(RNG, rho, MAX_TIME, SAMPLES)

    S = np.zeros(SAMPLES) # initialize array for price process
    nu = np.zeros(SAMPLES) # initialize array for volatility process 
    dt = MAX_TIME / (SAMPLES - 1)

    S[0] = S_0 
    nu[0] = nu_0

    # Update using Euler-Maruyama 
    for i in range(1, SAMPLES):
        # Since we take a sqrt we need to clip negative values whenever we do sqrt
        nu_abs = max(nu[i-1], 0.0)
        # We do this only in the update formula, not in the initial value 
        nu[i] = nu[i-1] + kappa * (theta - nu_abs) * dt + xi * np.sqrt(nu_abs) * (W_nu[i] - W_nu[i-1])

        S[i] = S[i-1] + mu * S[i-1] * dt + np.sqrt(nu_abs) * S[i-1] * (W_S[i] - W_S[i-1])

    return (t, S, nu)


def hestonSim_vec(rng, n, S_0, v_0, r, kappa, theta, xi, rho, T, steps):
    dt = T / steps
    Z = rng.standard_normal((steps, 2, n))
    Zv = Z[:, 0]
    Zs = rho * Z[:, 0] + np.sqrt(1 - rho**2) * Z[:, 1]
    S = np.empty((n, steps + 1)); S[:, 0] = S_0
    logS = np.full(n, np.log(S_0)); v = np.full(n, v_0)
    for k in range(steps):
        vp = np.maximum(v, 0.0); sq = np.sqrt(vp * dt)
        logS += (r - 0.5 * vp) * dt + sq * Zs[k]
        v = v + kappa * (theta - vp) * dt + xi * sq * Zv[k]
        S[:, k + 1] = np.exp(logS)
    return S

"""
def hestonSim_vec(rng, no_paths, S_0, nu_0, r, kappa, theta, xi, rho, MAX_TIME=1, SAMPLES=100):
    dt = MAX_TIME / SAMPLES
    Z = rng.standard_normal((SAMPLES, 2, no_paths))
    Zv = Z[:, 0]
    Zs = rho * Z[:, 0] + np.sqrt(1 - rho**2) * Z[:, 1]
    S = np.empty((no_paths, SAMPLES + 1)) 
    S[:, 0] = S_0
    logS = np.full(no_paths, np.log(S_0)) 
    nu_0s = np.full(no_paths, nu_0)
    
    for k in range(SAMPLES):
        nu_abs = np.maximum(nu_0s, 0.0); sq = np.sqrt(vp * dt)
        logS += (r - 0.5 * nu_abs) * dt + sq * Zs[k]
        nu = nu + kappa * (theta - nu_abs) * dt + xi * sq * Zv[k]
        S[:, k + 1] = np.exp(logS)
        
    return S
"""