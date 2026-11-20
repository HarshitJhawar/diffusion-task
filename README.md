# Guided sampling when a rule changes

A small diffusion model on a 2-D toy, steered at sampling time by rules it never saw in training. Every result is scored against the exact answer, which we can compute because the toy is 2-D.

- **Part A:** train the base model; build two distances; measure the noise floor and the base model's own error.
- **Part B:** guide the frozen model with rules A and B; sweep the guidance scale; test one design choice (does the gradient pass through the denoiser?).
- **Part C:** change rule A (tighten or reverse) where it lives in a guide or is baked into the model; fill a 2 × 2 table.

## One command

```bash
pip install -r requirements.txt
bash run_all.sh          # tests, Part A, Part B (sweep + design choice), Part C
```

Everything runs on a laptop CPU. Rough expected run times: Part A a few minutes; Part B and Part C tens of minutes each, mostly guided sampling with a gradient through the network.

## Workflow (order matters for the task's rules)

```bash
git init && git add . && git commit -m "Code for Parts A-C"

python -m pytest -q                 # 11 known-answer tests for the distances
python part_a.py                    # Part A
python part_b.py sweep              # Part B: exact target, noise floor, sweep over s

# Write HYPOTHESIS_B.md, commit it, then:
python part_b.py design             # gradient through the denoiser vs skipping it

# Write PREDICTIONS.md, commit it, then:
python part_c.py all                # baked model, four cells, repair of the worst cell
```

Commit your results after each step, too. `outputs/` is tracked by git, so the history shows the predictions came before the results. `checkpoints/` is not tracked.

## Diffusion formulation (the task asks us to state it)

Discrete-time DDPM (Ho et al. 2020):

- variance-preserving, 1000 steps, linear β schedule from 10⁻⁴ to 0.02;
- the network predicts the noise ε and is trained on L_simple;
- ancestral sampling with σ²_t = β̃_t.

The network has 3 hidden layers of 128 units, SiLU activations and a sinusoidal time embedding. It is trained for 20,000 Adam steps (learning rate 1e-3, cosine decay, batch 1024) on a fixed set of 100,000 points from p0. Samples come from an EMA copy of the weights (decay 0.999).

## Guidance (Part B), exactly

At each reverse step we compute the model's clean-sample estimate x̂₀ (DDPM Eq. 15), and the gradient

> g = ∇ with respect to x_t of [ log V_A(x̂₀) + log V_B(x̂₀) ]

**through the network** (autograd), as in DPS. Then we add it to the step's mean in the same way a score correction enters a DDPM step:

> mean ← mean + s · (β_t / √α_t) · g

So **s = 1 corresponds to exact Bayes** if the x̂₀ approximation were exact, and s > 1 over-weights the rules. Full reasoning is at the top of `src/guidance.py`. **This scaling is a choice; state it in the report.**

**Design choice tested** (`part_b.py design`): the same sampler with the gradient taken **through the denoiser**, vs **skipping it**, where the gradient of log V at x̂₀ is applied to x_t directly. This also answers the task's question "say whether your gradient passes through the denoiser, and why".

## Part C, exactly

- **Baked model:** Part A's code and seeds, training data filtered to x1 > 0.
- **The edit:** both cells end with guides A′ and B at the same s (by default, Part B's lowest-grid-distance s, which Part B saves in `outputs/choices.json`; override with `--s`). Every cell is scored against its new exact target, p0 × V_A′ × V_B, with its own noise floor.
- **Worst cell:** the one whose grid distance is furthest above its own noise floor.


## What is yours to write

- `PREDICTIONS.md`: predictions for the four cells, committed before any Part C run.
- `HYPOTHESIS_B.md`: the design-choice hypothesis, committed before that run.
- The explanation of the Part B sweep. `part_b.py` prints the facts: which s has the lowest distances, which has the most upper-right samples, and the exact upper-right share.
- The falsifiable claim for Part C, the two papers you found yourself, the report, and `TIMELOG.md` (including where you used an AI assistant).

## Outputs

Everything lands in one folder, `outputs/`. **`outputs/RESULTS.md` shows every table and every figure in one file**, in order Part A → B → C. Each script rebuilds it when it finishes.

| Command | Tables (in `outputs/`) | Figures (in `outputs/`) |
|---|---|---|
| `part_a.py` | `part_a.md`: noise floor, base-model error, training length, known-answer checks | `fig_a1` samples, `fig_a2` noise floor, `fig_a3` training length |
| `part_b.py sweep` | `part_b.md`: the sweep and mode shares; also saves the best s to `choices.json` | `fig_b1` distances and upper-right share vs s, `fig_b2` samples, `fig_b3` mode shares |
| `part_b.py design` | `part_b_design.md` | `fig_b4_design` |
| `part_c.py` | `part_c.md`: the 2 × 2 tables and mode shares; | `fig_c1` cells, `fig_c2` mode shares |

Checkpoints: `checkpoints/base_full.pt` (Part A) and `checkpoints/base_baked.pt` (Part C).

## Seeds

All seeds are in `src/config.py`:

- training data: 0
- network initialisation and batches: 0
- model sampling: 1, 2, 3 (every result)
- exact samples for noise floors: 100–119

Every guided result is the mean ± standard deviation over the three sampling seeds.

## Layout

```
src/config.py        shared settings and seeds
src/ground_truth.py  every exact answer: p0, grid, exact cell probabilities and mode shares,
                     and Target (p0 x rules: grid density, quadrant share, rejection sampler)
src/rules.py         rules A, B, A' (tighten, reverse) as soft scores, and their logs
src/diffusion.py     schedule, denoiser, training (Alg. 1), sampling (Alg. 2), save/load
src/guidance.py      guided sampler (frozen model, gradient at x0_hat, through the denoiser or not)
src/scoring.py       the two distances, grading against a Target, mean ± sd over seeds,
                     noise floors, running one setting over all seeds
tests/               known-answer tests for the distances
part_a.py, part_b.py, part_c.py, run_all.sh
```

## Tests (`python -m pytest -q`)

`tests/test_distances.py` checks the two distances and p0's exact values on cases with a known answer:

- identical inputs give 0;
- disjoint inputs give 1;
- all samples in one cell against a uniform target give 1 − 1/3600;
- samples from one bump against p0 give 7/8 on both distances;
- points outside the grid are counted;
- the closed-form and numerical cell probabilities agree;
- the exact mode shares of p0 are 1/8 each.

`part_a.py` repeats the main checks and writes them into `outputs/part_a.md`.
