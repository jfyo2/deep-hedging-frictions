import sys, os, math
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import importlib
import pandas as pd 
import torch 
import torch.nn as nn 

from src import gbm, pricer, heston, deltahedge 
from src.config import * 

def compute_cvar(pnl_dist, confidence_lvl):
    losses = -pnl_dist
    x = torch.quantile(losses.detach(), confidence_lvl)   # exact inner minimiser
    return x + torch.clamp(losses - x, min=0.0).mean() / (1 - confidence_lvl)


class CVaRLoss(nn.Module):
    def __init__(self, confidence_lvl):
        super().__init__()
        self.alpha = confidence_lvl 
        #self.x = nn.Parameter(torch.tensor(0.0))

    def forward(self, pnl_dist):
        return compute_cvar(pnl_dist, self.alpha)
        #return compute_cvar(pnl_dist, self.x, self.alpha)
        


class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, confidence_lvl, cost_rate, update_interval):
        super().__init__()

        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.alpha_lvl = confidence_lvl
        self.dt = update_interval 
        self.cost_rate = cost_rate 

        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.cells = nn.ModuleList(
            [nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
             for i in range(num_layers)]
        )

        path_update_interval = T / STEPS # Number of simulation Intervals 
        # path is updated
        self.steps_per_hedge = int(round(self.dt / path_update_interval)) # how many simulation 
        # grid-points one hedge interval spans

        assert abs(self.steps_per_hedge * path_update_interval - self.dt) < 1e-9 # for safety, check we have an integer
        

        
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def init_hidden_layer(self, batch_size):
        # initialization for both is exactly the same 
        h = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        return h, c

    # Given a stock price S_t, time until expiry tau_t, and delta value delta_t at time t, compute the delta value 
    # delta_{t+1} at the next time step. The hidden layer weights are also passed through this function 
    def step(self, S_t, tau_t, delta_t, hidden_layers):
    # Note: tau_t = T - t where T is the expiry time and t is current time. so
    # time tau_t represents time until expiry 
        
        # initialize parameters 
        h, c = hidden_layers
        # input consists of S_t, tau_t, and the previous delta_t, so we throw
        # these all into a single torch tensor.
        # We also normalize the parameters as in standard in ML. 
        x = torch.stack([torch.log(S_t / K), tau_t / T, delta_t], dim=1)

        # update hidden weights across all cells 
        for i, cell in enumerate(self.cells):
            h[i], c[i] = cell(x, (h[i], c[i]))
            x = h[i]

        # LSTM applies a linear transformation followed by a sigmoid to the
        # output 
        delta_tplus1 = torch.sigmoid(self.output_layer(x)).squeeze(-1)

        return delta_tplus1, (h, c)

    
    # We want to train the model on paths of different lengths, so we pass in a steps_to_maturity_array as well
    # Note: paths_array should be downsampled BEFORE inputting into this function so it matches 
    # the precision of the delta hedge 
    def simulate_process(self, paths_array, no_of_calls, steps_to_maturity_array):
        N_PATHS, N_POINTS = list(paths_array.shape)

        
        hidden = self.init_hidden_layer(N_PATHS) # initialize hidden layer 
        delta_ts = torch.zeros(N_PATHS) # we predict delta_t for EVERY path 

        process = [] # initialize list for storing process data 
        path_active_mask = []


        for i in range(N_POINTS):
            path_active = (i < steps_to_maturity_array) # used to ignore paths after they reach maturity. 
            # note that we simulate paths up to a fixed T, then mask them using this to model different maturity times 

            # vectorized step over all paths
            S_t = paths_array[:,i]
            intervals_remaining = steps_to_maturity_array - i 
            tau_t = (intervals_remaining * self.dt).clamp(min=1e-6) # tau_t represents time to maturity. clamp 
            # to avoid zero time to maturity as this breaks things 
            assert tau_t.max() <= T + 1e-6 # make sure time makes sense 
            
            delta_tplus1, hidden = self.step(S_t, tau_t, delta_ts, hidden)

            delta_ts = torch.where(path_active, delta_tplus1, delta_ts) # update delta_t -- frozen after expiry 

            process.append(delta_ts * no_of_calls)
            path_active_mask.append(path_active)

        # convert process list into torch tensor 
        process_tensor = torch.stack(process, dim=1)
        path_active_mask_tensor = torch.stack(path_active_mask, dim=1)

        return process_tensor, path_active_mask_tensor



    def trainModel(self, epochs, learn_rate, paths_per_maturity=30, path_generator=None, print_epochs=20):
        # path_generator = None defaults to vectorized Heston 
        # the batch size is paths_per_maturity * T / self.dt, where T is the maximum time a path can be simulated to
        #N_PATHS, N_POINTS = list(paths_array.shape)

        if path_generator is None:   # default to Heston paths 
            path_generator = lambda rng, n: heston.hestonSim(
                rng, n, S_0, NU_0, R, KAPPA, THETA, XI, RHO, T, STEPS)
            
        
        
        #model = LSTMmodel(3, 32, 1, 0.95) # instance of lstm 
        lossfn = CVaRLoss(self.alpha_lvl) # instance of CVaRLoss module 

        ss = np.random.SeedSequence(42)

        training_data = []

        # we can actually use a different learn rate for training self.x in the CVaR loss and the rest of the model if needed 
        # so we define these things separately 
        optimizer = torch.optim.Adam([
                                {'params': self.parameters(), 'lr': learn_rate}])
                                #{'params': lossfn.parameters(), 'lr': learn_rate}
                            #])


        max_hedge_intervals = int(round(T / self.dt)) # maximum number of samples in a path at the hedge precision level 
        # (not the underlying path precision level)
        maturity_times = self.dt * np.arange(1, max_hedge_intervals + 1) # we want to make sure the model can handle 
        # different times to maturity so we train on paths of different length

        # grid of possible option premiums for each possible maturity time 
        C_0_grid = torch.tensor([pricer.option_price_bsm(S_0, K, m, R, SIGMA) for m in maturity_times]).float()

        
        for epoch in range(1, epochs + 1):
            
            intervals_to_maturity_array = np.tile(np.arange(1, max_hedge_intervals + 1), paths_per_maturity)  
            epoch_rng = np.random.default_rng(ss.spawn(1)[0])
            
            paths_array = path_generator(epoch_rng, len(intervals_to_maturity_array))[1]  
            assert paths_array.shape[1] == STEPS + 1

            # we resample the paths array to match the precision of the hedge i.e. how often 
            # we're going to update the hedge 
            paths_array_resampled = torch.from_numpy(paths_array[:, ::self.steps_per_hedge]).float() 
            assert paths_array_resampled.shape[1] == max_hedge_intervals + 1
            
            intervals_to_maturity_array_torch = torch.from_numpy(intervals_to_maturity_array)

            # find the premium that matches the intervals to expiry
            # using the C_0 grid we computed earlier 
            # we need to subtract 1 because indices start at 0, of course 
            C_0_array = C_0_grid[intervals_to_maturity_array_torch - 1] 

            # simulate the process 
            hedge_process, path_expiry_mask = self.simulate_process(paths_array_resampled, CALLS_SOLD, 
                                                        intervals_to_maturity_array_torch)
            self.pnl_dist = deltahedge.hedge_pnl_torch(paths_array_resampled, hedge_process, path_expiry_mask, 
                                                  C_0_array, K, R, CALLS_SOLD, self.cost_rate, self.dt)
            
            # we define pnl_dist with self. so we can access it from outside; this will be useful later 
            loss = lossfn(self.pnl_dist) # run forward pass for losses 

            
            # update model based on above 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0) # gradient clipping to avoid exploding gradients
            optimizer.step()

            training_data.append({'Epoch': epoch, 'CVaR loss' : loss.item(), 'Mean P&L' : self.pnl_dist.mean().item()})

            
            if epoch % print_epochs == 0:
                print(f"epoch {epoch:4d}  CVaR loss: {loss.item():.4f}  "
                    f"mean P&L: {self.pnl_dist.mean().item():.4f}")
                    #f"x (VaR est.): {lossfn.x.item():.4f}")

        print("Training complete.") 

        # convert data to pandas dataframe 
        self.train_data = pd.DataFrame(training_data) 



    
    # Helper function to automatically convert numpy paths to PyTorch and resample 
    # before running simulate_process 
    def deep_hedge_on(self, paths_array, trace=False):
        # resample array 
        paths_array_resampled = torch.from_numpy(paths_array[:, ::self.steps_per_hedge]).float()  
    
        # guard to check shape fits
        max_hedge_intervals = int(round(T / self.dt))
        assert paths_array_resampled.shape[1] == max_hedge_intervals + 1
    
        # create empty mask matching the size of the path array 
        steps_to_maturity = torch.full((paths_array_resampled.shape[0],), max_hedge_intervals, dtype=torch.long)
        
        with torch.no_grad():
            hedge, mask = self.simulate_process(paths_array_resampled, CALLS_SOLD, steps_to_maturity)
    
    
        if trace:
            return hedge[:, :max_hedge_intervals], paths_array_resampled, mask 
        else: 
            return hedge[:, :max_hedge_intervals]     
