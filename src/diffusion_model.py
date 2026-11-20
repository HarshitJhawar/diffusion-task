"""
A small DDPM (Ho et al. 2020) for 2-D points.

Formulation used (the task asks us to name it):
  - Discrete-time DDPM, variance-preserving, T steps, linear beta schedule.
  - The network predicts the noise epsilon (DDPM Eq. 11 / Eq. 14, "L_simple").
  - Sampling: ancestral DDPM sampling (DDPM Algorithm 2), written in the equivalent "blend of x_t and the clean-sample estimate x0_hat" form. This form makes x0_hat explicit at every step, which is exactly where Part B's guidance will be taken.

Notation (same as the DDPM paper):
  beta_t       noise added at step t
  alpha_t      = 1 - beta_t
  abar_t       = alpha_1 * alpha_2 * ... * alpha_t
  x_t          = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps          (DDPM Eq. 4)
  x0_hat       = (x_t - sqrt(1 - abar_t) * eps_theta) / sqrt(abar_t)  (DDPM Eq. 15)
"""


import copy 
import math 
from dataclasses import dataclass 
import numpy as np 
import torch 
import torch.nn as nn 

# Create the linear beta schedule
@dataclass 
class Schedule: 
    T:int 
    beta: torch.Tensor
    alpha: torch.Tensor 
    abar: torch.Tensor 
    abar_prev: torch.Tensor   # abar_{t-1}, with abar_0 = 1  
    post_var: torch.Tensor # beta_tilde_t = (1 - abar_{t-1}) / (1 - abar_t) * beta_t  (DDPM Eq. 7)


def make_schedule(T=1000, beta_start=1e-4, beta_end=0.02): 
    """Linear beta schedule matching the DDPM paper.
    With T=1000 (steps), abar_T about 4e-5, x_T is pure noise
    """
    beta = torch.linspace(beta_start,beta_end,T,dtype=torch.float64)
    alpha = 1.0 - beta 
    abar = torch.cumprod(alpha,0) # cumulative product 
    abar_prev = torch.cat([torch.ones(1,dtype=torch.float64),abar[:-1]])
    post_var = (1 -abar_prev) / (1-abar)*beta 
    to32 = lambda v : v.float() # lambda function to convert the tensor dtype to float32 
    return Schedule(T,*(to32(v) for v in (beta,alpha,abar,abar_prev,post_var)))

def time_embedding(t,dim=32): 
    """Sinusoidal embedding of the step number."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

class Denoiser(nn.Module): 
    """3 hidden layers of `width` units with SiLU activations.

    Input : the noisy point x_t and time embedding of t.
    Output: a guess of the noise eps that was mixed into x_t.
    One network is shared across all T steps.
    """

    def __init__(self,width=128,t_emb_dim=32): 
        super().__init__()
        self.t_emb_dim = t_emb_dim 
        self.net = nn.Sequential(
            nn.Linear(2 + t_emb_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 2),
        )

    def forward(self,x,t): 
        return self.net(torch.cat([x,time_embedding(t,self.t_emb_dim)], dim=1))


def train(data,sched,steps=5000,batch=1024,lr=1e-3,ema_decay=0.999,
            seed=0, width=128, log_every=500):
    """Fit eps_theta to a fixed training data 

    Each step:
      1. pick a random batch of real points x0,
      2. pick a random step t for each point,
      3. draw noise eps and make x_t in one jump with DDPM Eq. 4,
      4. loss = MSE(eps, network's guess) # Since log p(x) is MSE 
      5. one Adam step on the network's weights.

    We also keep an EMA (exponential moving average) copy of the weights,
    a standard trick in diffusion models: averaging the weights over recent
    steps smooths out the jitter from noisy gradient steps. We sample from the
    EMA copy.

    Returns (ema_model, list of (step, loss)).
    """    
    torch.manual_seed(seed) 
    # We create generator to be used for training based on seed for reproducibility
    gen = torch.Generator().manual_seed(seed) 

    data_t = torch.as_tensor(data,dtype=torch.float32) 
    model = Denoiser(width=width)
    ema  = copy.deepcopy(model).requires_grad_(False) 
    opt = torch.optim.Adam(model.parameters(),lr=lr) 

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps) 
    sqrt_ab, sqrt_1mab = sched.abar.sqrt(), (1 - sched.abar).sqrt()
    log, running = [], 0.0

    # This is training step (Algorithm-1) following the DDPM paper 
    for step in range(1,steps+1): 
        idx = torch.randint(0,len(data_t),(batch,), generator=gen)
        x0 = data_t[idx] 
        t = torch.randint(0,sched.T, (batch,), generator=gen) 
        eps = torch.randn(x0.shape,generator=gen) 
        xt = sqrt_ab[t, None] * x0 + sqrt_1mab[t, None] * eps   
        loss = ((eps - model(xt,t)) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        lr_sched.step()

        with torch.no_grad():   # EMA update: ema = d * ema + (1 - d) * model
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(ema_decay).add_(pm, alpha=1 - ema_decay)

        running += loss.item() 
        if step % log_every== 0 or step == steps: 
            log.append((step,running/log_every))
            running = 0.0 
    return ema,log 

# Sampling (DDPM Algorithm 2)
def predict_x0 (model,sched,xt,t_idx):
    """The model's current estimate of the clean sample 
    Tweedie's formula in the DPS paper. Returns (x0_hat, eps_hat).
    This method is utilised in sample method below 
    """
    t = torch.full((len(xt),),t_idx,dtype=torch.long)
    eps_hat = model(xt,t) 
    x0_hat = (xt - (1 - sched.abar[t_idx]).sqrt() * eps_hat) / sched.abar[t_idx].sqrt()
    return x0_hat, eps_hat

@torch.no_grad()
def sample(model, sched, n, seed):
    """Unguided sampling. 

    For t = T, ..., 1:
        x0_hat   = model's clean-sample estimate                   
        mean     = c0 * x0_hat + ct * x_t                          
        x_{t-1}  = mean + sqrt(beta_tilde_t) * z,  z ~ N(0, I)     
    """

    gen = torch.Generator().manual_seed(seed) 
    x = torch.randn((n,2),generator=gen)        # x_T ~ N(0, I), ie, start from pure noise 
    for i in reversed(range(sched.T)): 
        x0_hat,_ = predict_x0(model,sched,x,i)
        c0 = sched.abar_prev[i].sqrt() * sched.beta[i] / (1 - sched.abar[i])
        ct = sched.alpha[i].sqrt() * (1 - sched.abar_prev[i]) / (1 - sched.abar[i])
        mean = c0 * x0_hat + ct * x     
        # For i>0 step, we add tiny-bit fresh noise for better sampling and for last step, we don't add noise (as followed in Algorithm 2 of DDPM Paper)  
        if i > 0:
            x = mean + sched.post_var[i].sqrt() * torch.randn(x.shape, generator=gen)
        else:
            x = mean
    return x.numpy().astype(np.float64)       


def save_model(model,path,**info): 
    """Save weights plus a few facts about how the model was trained."""
    torch.save({"state_dict": model.state_dict(), **info}, path)    

def load_model(path): 
    """ 
    Load a model saved by Part A or Part C. 
    """
    ckpt = torch.load(path, map_location="cpu")
    model = Denoiser()
    model.load_state_dict(ckpt["state_dict"])
    model.eval().requires_grad_(False)
    info = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, info