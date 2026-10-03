import numpy as np 

# All random processes in this notebook must be fed with a random number 
# generator 'RNG' which is of the form np.random.default_rng(s) for some seed s


def brownianMotion(RNG, MAX_TIME=1, SAMPLES=100):
    """
    # Generates a 1D standard Brownian motion with included time indices: outputs (t,B) 
    # where t is an array of time points and B is an array of the corresponding 
    # values of the Brownian motion at the time points in t
    """
    if SAMPLES <= 1:
        raise ValueError("The number of samples must be an integer greater than or equal to 2.")

    # create a grid of normal variables
    Z = RNG.normal(0.0, 1.0, SAMPLES) 

    # Brownian paths have jumps B_{t+s} - B_s ~ \sqrt{t} Z, so we can implement as follows:
    t = np.linspace(0, MAX_TIME, SAMPLES)
    dt = MAX_TIME / (SAMPLES - 1) 
    BM = np.zeros(SAMPLES)
    BM[0] = 0
    for i in range(1, SAMPLES):
        BM[i] = BM[i-1] + np.sqrt(dt) * Z[i]

    return (t, BM)





def geometricBrownianMotion(RNG, S_0, mu, sigma, MAX_TIME=1, SAMPLES=100):
    """
    # Monte Carlo simulation of geometric Brownian motion with 
    # initial value S_0, drift mu, diffusion constant sigma 
    # in the time interval [0, MAX_TIME] 
    """
    # We use the analytic solution due to Ito: 
    # S_t = S_0 exp((\mu - \sigma^2/2) t + \sigma W_t)

    # Generate Brownian motion 
    (t, B) = brownianMotion(RNG, MAX_TIME, SAMPLES)

    # Plug into vectorized GBM formula 
    GBM = S_0 * np.exp((mu - sigma**2 / 2) * t + sigma * B)

    return (t, GBM)
