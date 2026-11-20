"""
Part A: build a trained sampler and.

What this script does, in order:
  1. Checks the two distances on cases with a known answer (the same checks
     as tests/test_distances.py.
  2. Measures the NOISE FLOOR: 20,000 exact samples from p0 scored against the
     exact p0. Repeated over 20 seeds, plus a curve versus
     sample count to show it shrinks like 1/sqrt(N).
  3. Trains the base diffusion model on samples from p0 and saves it.
  4. Scores the unguided base model against p0 (3 sampling seeds). The gap
     between this and the noise floor is the error the base model brings to
     every later part.
  5. A small check of how that error depends on the number of training steps.

"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import ground_truth
from src import diffusion_model as D
from src.scoring import (grid_distance, grid_histogram, mean_sd, mode_share_distance,
                         mode_shares)

# Fixed settings and seeds 
from src.config import (N_EVAL, N_TRAIN_DATA, DATA_SEED, TRAIN_SEED, SAMPLE_SEEDS,
                        FLOOR_SEEDS, T_STEPS, TRAIN_STEPS, OUTPUTS,
                        BASE_FULL_CKPT, make_dirs, write_section)

STEP_CHECK = [2_000, 5_000, 10_000] 
make_dirs()

def score(samples, masses, outside, shares):
    return grid_distance(samples, masses, outside), mode_share_distance(samples, shares)


def main():
    t_start = time.time()
    results = {}

    # quantities for p0 
    masses, outside = ground_truth.exact_grid_masses_p0()
    shares = ground_truth.EXACT_MODE_SHARES_P0
    results["exact_p0"] = {"mass_outside_grid": float(outside),
                           "mode_shares": shares.tolist()}

    # ---- 1. known-answer checks of the distances ---------------------------
    rng = np.random.default_rng(1)
    one_mode = ground_truth.CENTRES[0] + ground_truth.STD * rng.standard_normal((200_000, 2))
    numeric, _ = ground_truth.numerical_grid_masses(ground_truth.density_p0)
    frac, out = grid_histogram(one_mode)
    checks = {
        "grid: target equals the sample histogram (expect 0)":
            grid_distance(one_mode, frac, out),
        "grid: one mode only vs p0 (expect 7/8 = 0.875)":
            grid_distance(one_mode, masses, outside),
        "mode share: one mode only vs p0 (expect 7/8 = 0.875)":
            mode_share_distance(one_mode, shares),
        "exact masses: closed form vs numerical integration, TV (expect ~0)":
            0.5 * np.abs(masses - numeric).sum(),
        "exact mode shares by integration, max error vs 1/8 (expect ~0)":
            np.abs(ground_truth.exact_mode_shares(ground_truth.density_p0) - 1 / 8).max(),
    }
    results["known_answer_checks"] = {k: float(v) for k, v in checks.items()}
    print("1. Known-answer checks")
    for k, v in checks.items():
        print(f"   {v:.5f}   {k}")

    # 2. noise floor 
    floor = [score(ground_truth.sample_p0(N_EVAL, np.random.default_rng(s)), masses, outside, shares)
             for s in FLOOR_SEEDS]
    results["noise_floor_N20000"] = {"grid": mean_sd([f[0] for f in floor]),
                                     "mode_share": mean_sd([f[1] for f in floor])}
    # how the floor changes with sample count (10 seeds per size)
    sizes = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]

    curve = {"n": sizes, "grid": [], "mode_share": []}
    for n in sizes:
        sc = [score(ground_truth.sample_p0(n, np.random.default_rng(1000 + s)), masses, outside, shares)
              for s in range(10)]
        curve["grid"].append(mean_sd([a for a, _ in sc]))
        curve["mode_share"].append(mean_sd([b for _, b in sc]))
    results["noise_floor_vs_n"] = curve
    nf = results["noise_floor_N20000"]
    print(f"2. Noise floor at N={N_EVAL} over {len(FLOOR_SEEDS)} seeds: "
          f"grid {nf['grid']['mean']:.4f} ± {nf['grid']['sd']:.4f}, "
          f"mode share {nf['mode_share']['mean']:.4f} ± {nf['mode_share']['sd']:.4f}")

    # 3. train the base model 
    data = ground_truth.sample_p0(N_TRAIN_DATA, np.random.default_rng(DATA_SEED))
    sched = D.make_schedule(T_STEPS)
    t0 = time.time()
    model, loss_log = D.train(data, sched, steps=TRAIN_STEPS, seed=TRAIN_SEED)
    train_time = time.time() - t0
    D.save_model(model, BASE_FULL_CKPT, T=T_STEPS, train_steps=TRAIN_STEPS,
                 data_seed=DATA_SEED, train_seed=TRAIN_SEED,
                 training_data="all of p0", seconds=train_time)
    results["training"] = {"steps": TRAIN_STEPS, "seconds": train_time,
                           "loss_log": loss_log}
    print(f"3. Trained base model: {TRAIN_STEPS} steps in {train_time:.0f}s, "
          f"final loss {loss_log[-1][1]:.4f}")

    # 4. unguided base model vs p0 
    t0 = time.time()
    model_samples = {s: D.sample(model, sched, N_EVAL, seed=s) for s in SAMPLE_SEEDS}
    sample_time = (time.time() - t0) / len(SAMPLE_SEEDS)
    sc = [score(x, masses, outside, shares) for x in model_samples.values()]
    results["unguided_base_vs_p0"] = {
        "grid": mean_sd([a for a, _ in sc]), "mode_share": mean_sd([b for _, b in sc]),
        "mode_shares_per_seed": [mode_shares(x).tolist() for x in model_samples.values()],
        "seconds_per_20000_samples": sample_time}
    ub = results["unguided_base_vs_p0"]
    print(f"4. Unguided base model over {len(SAMPLE_SEEDS)} seeds: "
          f"grid {ub['grid']['mean']:.4f} ± {ub['grid']['sd']:.4f}, "
          f"mode share {ub['mode_share']['mean']:.4f} ± {ub['mode_share']['sd']:.4f}")

    # 5. base-model error versus training length 
    step_check = {"steps": [], "grid": [], "mode_share": []}
    for steps in STEP_CHECK + [TRAIN_STEPS]:
        if steps == TRAIN_STEPS:
            g, m = ub["grid"], ub["mode_share"]
        else:
            m_short, _ = D.train(data, sched, steps=steps, seed=TRAIN_SEED)
            s_short = [score(D.sample(m_short, sched, N_EVAL, seed=s), masses, outside, shares)
                       for s in SAMPLE_SEEDS]
            g, m = mean_sd([a for a, _ in s_short]), mean_sd([b for _, b in s_short])
        step_check["steps"].append(steps)
        step_check["grid"].append(g)
        step_check["mode_share"].append(m)
        print(f"5. Training steps {steps:>6}: grid {g['mean']:.4f} ± {g['sd']:.4f}, "
              f"mode share {m['mean']:.4f} ± {m['sd']:.4f}")
    results["error_vs_training_steps"] = step_check

    results["total_seconds"] = time.time() - t_start
    write_markdown(results)
    make_figures(results, masses, model_samples[SAMPLE_SEEDS[0]])
    print(f"Done in {results['total_seconds']:.0f}s. See outputs/RESULTS.md.")


# Reporting
def write_markdown(r):
    nf, ub = r["noise_floor_N20000"], r["unguided_base_vs_p0"]
    f = lambda d: f"{d['mean']:.4f} ± {d['sd']:.4f}"
    lines = [
        "# Part A results", "",
        f"All distances use {N_EVAL:,} samples. ± is one standard deviation across seeds.", "",
        "| What is scored against exact p0 | Seeds | Grid distance | Mode-share distance |",
        "|---|---|---|---|",
        f"| Exact samples (noise floor) | {len(FLOOR_SEEDS)} | {f(nf['grid'])} | {f(nf['mode_share'])} |",
        f"| Unguided base model ({TRAIN_STEPS:,} training steps) | {len(SAMPLE_SEEDS)} | {f(ub['grid'])} | {f(ub['mode_share'])} |",
        "", "## Base-model error versus training length", "",
        "| Training steps | Grid distance | Mode-share distance |", "|---|---|---|",
    ]
    sc = r["error_vs_training_steps"]
    for s, g, m in zip(sc["steps"], sc["grid"], sc["mode_share"]):
        lines.append(f"| {s:,} | {f(g)} | {f(m)} |")
    lines += ["", "## Known-answer checks", "", "| Check | Value |", "|---|---|"]
    for k, v in r["known_answer_checks"].items():
        lines.append(f"| {k} | {v:.5f} |")
    write_section("part_a", lines, ["fig_a1_base_model_samples.png", "fig_a2_noise_floor.png",
                                    "fig_a3_training_length.png"])


def make_figures(r, masses, model_x):
    # Figure A1: exact density, exact samples, model samples, on one colour scale
    exact_x = ground_truth.sample_p0(N_EVAL, np.random.default_rng(FLOOR_SEEDS[0]))
    panels = [("Exact base distribution\n(probability of each grid cell)", masses),
              (f"{N_EVAL:,} exact samples\n(fraction in each cell)", grid_histogram(exact_x)[0]),
              (f"{N_EVAL:,} samples from the trained model\n(fraction in each cell)",
               grid_histogram(model_x)[0])]
    vmax = max(p.max() for _, p in panels)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for ax, (title, img) in zip(axes, panels):
        im = ax.imshow(img.T, origin="lower", extent=[-3, 3, -3, 3], cmap="viridis",
                       vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("first coordinate")
        ax.set_ylabel("second coordinate")
    fig.colorbar(im, ax=axes, shrink=0.85, label="probability per grid cell")
    fig.suptitle("The base data and what the trained model generates", fontsize=13)
    fig.savefig(OUTPUTS / "fig_a1_base_model_samples.png", dpi=150)
    plt.close(fig)

    # Figure A2: noise floor versus sample count, with the model's score marked
    c, ub = r["noise_floor_vs_n"], r["unguided_base_vs_p0"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, key, name in [(axes[0], "grid", "Grid distance"),
                          (axes[1], "mode_share", "Mode-share distance")]:
        mean = np.array([d["mean"] for d in c[key]])
        sd = np.array([d["sd"] for d in c[key]])
        ax.errorbar(c["n"], mean, yerr=sd, marker="o", capsize=3,
                    label="exact samples (best any sampler can do)")
        ax.errorbar([N_EVAL], [ub[key]["mean"]], yerr=[ub[key]["sd"]], marker="s",
                    color="C3", capsize=3, linestyle="none",
                    label="trained model (3 sampling seeds)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("number of samples")
        ax.set_ylabel(f"{name.lower()} to the exact distribution")
        ax.set_title(f"{name}: noise floor versus the trained model")
        ax.axvline(N_EVAL, color="gray", lw=0.8, ls=":")
        ax.legend(fontsize=9)
    fig.savefig(OUTPUTS / "fig_a2_noise_floor.png", dpi=150)
    plt.close(fig)

    # Figure A3: base-model error versus training steps, floor as a band
    sc, nf = r["error_vs_training_steps"], r["noise_floor_N20000"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, key, name in [(axes[0], "grid", "Grid distance"),
                          (axes[1], "mode_share", "Mode-share distance")]:
        mean = [d["mean"] for d in sc[key]]
        sd = [d["sd"] for d in sc[key]]
        ax.errorbar(sc["steps"], mean, yerr=sd, marker="o", capsize=3,
                    label="trained model (3 sampling seeds)")
        lo = nf[key]["mean"] - nf[key]["sd"]; hi = nf[key]["mean"] + nf[key]["sd"]
        ax.axhspan(lo, hi, color="gray", alpha=0.3, label="noise floor (exact samples)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("training steps")
        ax.set_ylabel(f"{name.lower()} to the exact distribution")
        ax.set_title(f"{name} of the base model versus training length")
        ax.legend(fontsize=9)
    fig.savefig(OUTPUTS / "fig_a3_training_length.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
