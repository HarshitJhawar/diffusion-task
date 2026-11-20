"""
Guided sampling: steer a FROZEN base diffusion model with rules it never saw.

Sample from  p*(x) ∝ p0(x) V(x),  where V(x) = V_A(x) V_B(x) (or any rules).
In score form (Bayes, guide Part F.4), the score we would like at noise
level t is
    ∇ log p*_t(x_t) = ∇ log p0_t(x_t)  +  ∇ log V_t(x_t)
                      [base model]        [guidance: unknown]

----------------------------------------------------
We need to add s times the gradient at each reverse step. Adding s*g
in raw form would make the total push depend on how many steps we take
(1000 steps means 1000 pushes). Instead we add it the way a score enters a
DDPM step. DDPM's mean can be written with the score s_theta = -eps/sqrt(1-abar):
    mean = ( x_t + beta_t * s_theta ) / sqrt(alpha_t)          (guide Part E.13)
so adding a correction g to the score adds  (beta_t / sqrt(alpha_t)) * g  to
the mean. We therefore use
    mean_guided = mean + s * (beta_t / sqrt(alpha_t)) * g.
With this scaling, s = 1 is "plain Bayes": if the x0_hat approximation were
exact, s = 1 would sample p* exactly. s > 1 over-weights the rules, which is
like using V^s (Dhariwal Sec. 4.3). State this interpretation in the report:
it is a choice.

The one design switch: grad_through_denoiser (default True)
-----------------------------------------------------------
    True : g = ∇_{x_t} log V(x0_hat(x_t)), with the chain rule going back
           THROUGH the network (autograd). This is the honest gradient of
           what we wrote down, as in DPS. It knows that at high noise, moving
           x_t barely moves x0_hat (the network mostly predicts the average
           of the data), so the push there is small.
    False: g = ∇ log V evaluated at x0_hat, applied to x_t directly, as if
           x0_hat moved one-for-one with x_t. No backward pass through the
           network: cheaper, but it ignores how x0_hat responds to x_t, so
           it may push too hard early on.
The task asks "say whether your gradient passes through the denoiser, and
why". Part B's design-choice experiment compares the two.

The sampler draws random numbers in exactly the same order as the Part A
sampler, so with s = 0 it reproduces diffusion.sample() for the same seed.
The tests check this.
"""


import numpy as np 
import torch 
from .rules import combined_log_score 

def _guidance_terms(model,x,i,abar_t,rules,s,grad_through_denoiser): 
    """
    Returns (eps_hat, x0_hat, g), detached from autograd.
    g is zero when there is no guidance (s = 0 or no rules).
    """

    t = torch.full((x.shape[0],), i, dtype=torch.long)
    sqrt_ab, sqrt_1mab = abar_t.sqrt(), (1 - abar_t).sqrt()

    if s == 0 or not rules:
        with torch.no_grad():
            eps = model(x, t)
            x0_hat = (x - sqrt_1mab * eps) / sqrt_ab
        return eps, x0_hat, torch.zeros_like(x)

    if grad_through_denoiser:
        # DPS-style: differentiate log V(x0_hat(x_t)) with respect to x_t,
        x_leaf = x.detach().requires_grad_(True)
        eps = model(x_leaf, t)
        x0_hat = (x_leaf - sqrt_1mab * eps) / sqrt_ab
        g = torch.autograd.grad(combined_log_score(rules, x0_hat).sum(), x_leaf)[0]
        return eps.detach(), x0_hat.detach(), g

    with torch.no_grad(): 
        eps = model(x,t) 
        x0_hat = (x - sqrt_1mab * eps) / sqrt_ab 

    x0_leaf = x0_hat.detach().requires_grad_(True)
    g = torch.autograd.grad(combined_log_score(rules,x0_leaf).sum(), x0_leaf)[0] 
    return eps, x0_hat, g

def guided_sample(model,sched,n,seed,rules=(),s=0.0, grad_through_denoiser=True): 
    """Draw n samples from the frozen model, guided by `rules` with scale s.

    DDPM sampling (Algorithm 2) plus the guidance term. For t = T..1:
        x0_hat, g = clean estimate and guidance gradient       (see above)
        mean      = c0 * x0_hat + ct * x_t                     (, x0 -> x0_hat)
                    + s * beta_t / sqrt(alpha_t) * g           (guidance)

        x_{t-1}   = mean + sqrt(beta_tilde_t) * z              (no noise at the last step)
    """

    for p in model.parameters(): 
        p.requires_grad_(False)

    rules = list(rules) 
    gen = torch.Generator().manual_seed(seed) 
    x = torch.randn((n,2),generator=gen) 

    for i in reversed(range(sched.T)): 
        eps,x0_hat, g= _guidance_terms(model, x, i, sched.abar[i], rules, s,            grad_through_denoiser)

        with torch.no_grad():
            c0 = sched.abar_prev[i].sqrt() * sched.beta[i] / (1 - sched.abar[i])
            ct = sched.alpha[i].sqrt() * (1 - sched.abar_prev[i]) / (1 - sched.abar[i])
            mean = c0 * x0_hat + ct * x
            # Guidance: a correction to the score, scaled as a score enters the mean.
            mean = mean + s * sched.beta[i] / sched.alpha[i].sqrt() * g
            if i > 0:
                x = mean + sched.post_var[i].sqrt() * torch.randn(x.shape, generator=gen)
            else:
                x = mean  # last step: no added noise

    return x.numpy().astype(np.float64)