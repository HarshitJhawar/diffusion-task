# Guided sampling when a rule changes

A small diffusion model, guided at sampling time. Every result is scored against the exact answer. 

- **Part A:** train the base model; build two total variational distances; measure the noise floor and the base model's own error.
- **Part B:** guide the frozen model with rules A and B; sweep the guidance scale; we test one design choice (does the gradient pass through the denoiser?).
- **Part C:** change rule A (tighten/ reverse) "in guide" or baked into model. 

## One command for running 

```bash
pip install -r requirements.txt
bash run_all.sh          # tests, Part A, Part B (sweep + design choice), Part C
```


## Diffusion formulation 
Discrete-time DDPM (Ho et al. 2020):
- variance-preserving, 1000 steps, linear β schedule from 10⁻⁴ to 0.02;
- the network predicts the noise ε and is trained on L_simple;
- ancestral sampling (Algorithm 2) with σ²_t = β̃_t.

The network has 3 hidden layers of 128 units, SiLU activations with sinusoidal time embedding. Trained over 20,000 Adam steps (lr=1e-3, cosine decay, batch 1024) on set of 100,000 points from p0. 

## Guidance (Part B), exactly
At each reverse step we compute the model's clean-sample estimate x̂₀ (DDPM Eq. 15), and the gradient

> g = ∇ with respect to x_t of [ log V_A(x̂₀) + log V_B(x̂₀) ]

**through the network** (autograd), as in DPS. Then we add it to the step's mean in the same way a score correction enters a DDPM step:

> mean ← mean + s · (β_t / √α_t) · g

**Design choice tested** (`part_b.py design`): the same sampler with the gradient taken **through the denoiser**, vs **skipping it**, where the gradient of log V at x̂₀ is applied to x_t directly.

## Part C, exactly
- **Baked model:** Based on Part A's code and seeds, training data filtered to x1 > 0.

Every output in one folder, `outputs/`. **`outputs/RESULTS.md` 
All seeds are in `src/config.py`. 

## Layout

```
src/config.py            settings, seeds, output helpers
src/ground_truth.py      p0, the grid, exact cell probabilities and mode shares, exact targets
src/rules.py             rules A, B, A' as soft scores
src/diffusion_model.py   schedule, denoiser, training, sampling
src/guidance.py          guided sampler
src/scoring.py           the two distances, scoring over seeds, noise floors
tests/                   known-answer tests for the distances
part_a.py, part_b.py, part_c.py, run_all.sh
```