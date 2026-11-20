# Part B design choice: hypothesis

- Design choice I am testing: whether the guidance gradient passes through the denoiser (gradient of log V(x0_hat(x_t)) through the network, vs the gradient of log V at x0_hat used directly)
- Guidance scales I will compare at: 1, 2, 4

## Hypothesis

At the same guidance scale, passing the gradient through the denoiser will push more samples into the two upper-right modes. So "through" will have a higher upper-right share and a larger mode-share distance, and the gap will grow with s.
