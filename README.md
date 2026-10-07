# Deep Hedging under Market Frictions

## Introduction

A PyTorch implementation of **deep hedging** (Buehler et al., 2019) for a market maker who has sold European call options and rebalances a stock position under proportional transaction costs. We train an LSTM model to minimize the CVaR of terminal hedging P&L, and compare the results to standard Black-Scholes delta hedging under both GBM and Heston asset price path dynamics. 

The aim is to see if 
1. A small LSTM can outperform standard delta hedging on Heston paths under transaction costs.
2. With no frictions on GBM, the learned hedge can recover the Black-Scholes delta.

AI tool disclosure: Claude Sonnet 5.5 was used to plan the project, fix bugs, standardize conventions, and draft new code in a number of cases. However, completed code, as well as the commentary in main.ipynb, is almost all human-written. Code in sections marked with WIP may be largely AI generated; these are drafts and should not be taken as finalized. 


## Repository Layout 

The main file in which all results are compiled with commentary is notebooks/main.ipynb. People viewing this project should start there. 

```
├── src/
│   ├── config.py        # global parameters 
│   ├── gbm.py           # vectorised GBM path generator (Numba)
│   ├── heston.py        # vectorised Heston path generator (Numba)
│   ├── pricer.py        # Black-Scholes price and Monte Carlo price calculators 
│   ├── deltahedge.py    # BS delta, baseline delta-hedge P&L (NumPy), differentiable cash-account P&L (PyTorch)
│   └── deephedging.py   # CVaR loss, LSTM policy (LSTMmodel), rollout (simulate_process), training (trainModel)
├── notebooks/
│   ├── main.ipynb                  # end-to-end experiments and evaluation
│   │   main-nocommentary.ipynb     # main notebook with commentary removed 
│   └── bugfix commentary.ipynb     # narrative of bugs found and how they were diagnosed
├── LICENSE
└── README.md
```

## Getting Started 

To install dependencies, run:

```
git clone https://github.com/jfyo2/deep-hedging-frictions
cd deep-hedging-frictions
pip install numpy scipy pandas matplotlib torch "numba>=0.56" jupyter
jupyter notebook
```
Then enter notebooks/main.ipynb and run the notebook. 

### GPU support? 

The current version of the code does not support CUDA GPU speedup, though we have implemented numba compilation for key functions to speed up path generation. As of writing this I only have access to an older laptop with a mediocre processor, and it takes sub 5 mins to train the model on CPU, so I believe training time should not be an issue even without CUDA. 

### Train and evaluate a deep hedger on Heston paths:

Here's a quick code example to show how to train and evaluate the model using CVaR: 

```
import numpy as np
import torch
from src import heston, pricer, deltahedge, deephedging
from src.config import *

COST_RATE = 0.001

# Train the model 
torch.manual_seed(42)
model = deephedging.LSTMmodel(3, 32, 1, 0.95, COST_RATE, UPDATE_INTERVAL)   # input, hidden, layers, alpha, cost, dt
model.trainModel(epochs=1500, learn_rate=0.002, paths_per_maturity=30)       # defaults to Heston training paths

# Generate test paths 
rng = np.random.default_rng(999)
_, paths, _ = heston.hestonSim(rng, 100_000, S_0, NU_0, R, KAPPA, THETA, XI, RHO, T, STEPS)

# Baseline: Black-Scholes delta hedging 
C_0 = pricer.option_price_bsm(S_0, K, T, R, SIGMA)
bs_pnl = deltahedge.delta_hedge_pnl(paths, S_0, C_0, K, R, SIGMA, CALLS_SOLD, COST_RATE, T, UPDATE_INTERVAL)

# Deep hedge on the same paths 
hedge, prices, mask = model.deep_hedge_on(paths, trace=True)   # note prices is resampled to the hedge grid
C_0_array = torch.full((prices.shape[0],), float(C_0))
deep_pnl = deltahedge.hedge_pnl_torch(prices, hedge, mask, C_0_array, K, R,
                                      CALLS_SOLD, COST_RATE, model.dt).numpy()

def cvar_np(pnl, alpha=0.95):
    losses = -np.asarray(pnl)
    q = np.quantile(losses, alpha)
    return q + np.mean(np.maximum(losses - q, 0.0)) / (1 - alpha)

for a in (0.75, 0.9, 0.95, 0.99):
    print(f"alpha={a}:  BS delta {cvar_np(bs_pnl, a):7.1f}   deep hedge {cvar_np(deep_pnl, a):7.1f}")
```

## License

This project is released under the MIT License.