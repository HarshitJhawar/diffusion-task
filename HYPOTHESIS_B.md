# Part B design choice: hypothesis

- Design choice I am testing: whether the guidance gradient passes through the denoiser (gradient of log V(x0_hat(x_t)) through the network, vs the gradient of log V at x0_hat used directly)
- Guidance scales I will compare at:  1, 2, 4

## Hypothesis
At the same guidance scale, passing the gradient through the denoiser will push more samples into the two upper-right modes. So "through" will have a higher upper-right share and a larger mode-share distance, and the gap will grow with s.


## Why
With the gradient through the denoiser, the push involves how much the clean-sample
guess x0_hat moves when the noisy point moves. When a noisy point sits between two
bumps, a small move can switch which bump the guess points to, so x0_hat moves a lot
and the chain rule multiplies the push. These large pushes near the boundaries move
samples into the bumps the rules prefer. Skipping the denoiser treats x0_hat as moving
one-for-one with the noisy point, so it never gets this boost.