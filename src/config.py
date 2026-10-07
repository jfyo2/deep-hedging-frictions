"""
(c) jfyo2 2026. This project is licensed under the MIT License. 

This config file defines the values of important global variables. 
"""

# we will use capital letters for constant instances of the parameters 
# and lowercase for these parameters where they appear in functions

S_0 = 75 # Initial asset price 
R = 0.05 # Annual risk-free rate 
SIGMA = 0.25 # Annual volatility -- on the high end, because typically more delta movement makes things more interesting 
T = 1.0 # Time to maturity 
STEPS = 100 # Number of time steps 

K = 75 # Typically we take the strike price to equal S_0 in hedging studies 


# Semi-realistic Greeks
KAPPA = 2.0 # mean-reversion speed 
THETA = SIGMA**2 # Long-run variance 
NU_0 = SIGMA**2 # initial variance 
XI = 0.4 # vol-of-vol 
RHO = -0.7 # Price-vol correlation 



UPDATE_INTERVAL = 0.02 # how often the market maker rebalances their hedge 
CALLS_SOLD = 100 # number of calls the market maker sells 