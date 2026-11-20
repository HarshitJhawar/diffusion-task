"""
Part B: how close does a FROZEN base model with two guides get to the exact target?

    Target   p*(x) ∝ p0(x) * V_A(x) * V_B(x)
             V_A(x) = sigmoid(x1 / 0.5)   "x1 > 0"
             V_B(x) = sigmoid(x2 / 0.5)   "x2 > 0"
    Method   at each reverse step, add s times the gradient of log V_A + log V_B, taken at the model's clean-sample estimate x0_hat 

  python part_b.py sweep
      1. Computes the exact target twice: as a grid density, and samples by rejection. Compare them and measures the noise floor for p*. 
      2. Sweeps the guidance scale s over S_VALUES, 3 seeds each, with the gradient passing through the denoiser. 
  python part_b.py design
      The design-choice experiment: gradient THROUGH the denoiser vs SKIPPING it, at a few scales, 3 seeds each.
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import diffusion_model as D
from src import ground_truth
from src.config import (BASE_FULL_CKPT, FLOOR_SEEDS, N_EVAL, OUTPUTS, SAMPLE_SEEDS,
                        T_STEPS, make_dirs, save_choice, write_section)
from src.ground_truth import Target
from src.rules import RULE_A, RULE_B
from src.scoring import best_by, fmt, histogram_image, noise_floor, run_setting

# Settings
RULES = [RULE_A, RULE_B]
S_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]       # five different values from 0 to 8
DESIGN_S_VALUES = [1.0, 2.0, 4.0]               # scales for the design-choice comparison
DESIGN_VARIANTS = [("gradient through the denoiser", True),
                   ("gradient skips the denoiser", False)]

def load_base():
    if not BASE_FULL_CKPT.exists():
        raise SystemExit(f"{BASE_FULL_CKPT} not found. Run `python part_a.py` first.")
    model, info = D.load_model(BASE_FULL_CKPT)
    return model, D.make_schedule(info.get("T", T_STEPS))


def floor_band(ax, floor_entry, label="noise floor (exact samples)"):
    """Grey band at the noise floor: mean ± one standard deviation."""
    lo = floor_entry["mean"] - floor_entry["sd"]
    hi = floor_entry["mean"] + floor_entry["sd"]
    ax.axhspan(lo, hi, color="gray", alpha=0.3, label=label)


# Command 1: the sweep
def sweep():
    make_dirs()
    t_start = time.time()
    model, sched = load_base()

    # 1. the exact target, twice 
    print("1. Exact target p* ∝ p0 * V_A * V_B")
    target = Target(RULES, name="p* = p0 reweighted by rules A and B")
    # Route 1 (density) gives the cell probabilities and shares.
    # Route 2 (rejection samples) scored against route 1 gives the noise floor.
    # If the two routes agree, that floor looks like Part A's (a few hundredths).
    floor = noise_floor(target, FLOOR_SEEDS, N_EVAL)
    print(f"   exact upper-right share {target.upper_right_share:.4f}")
    print(f"   exact mode shares {np.round(target.mode_shares, 4).tolist()}")
    print(f"   noise floor (rejection samples vs density, {len(FLOOR_SEEDS)} seeds): "
          f"grid {fmt(floor['grid'])}, mode share {fmt(floor['mode_share'])}")

    # 2. sweep the scale s 
    print(f"2. Sweep of s over {S_VALUES}, seeds {SAMPLE_SEEDS}")
    runs = []
    for s in S_VALUES:
        r = run_setting(model, sched, target, RULES, s, keep_samples=True)
        runs.append(r)
        m = r["summary"]
        print(f"   s = {s:>4}: grid {fmt(m['grid'])}  mode share {fmt(m['mode_share'])}  "
              f"upper-right {fmt(m['upper_right'], 3)}  non-finite {m['non_finite']['mean']:.0f}  "
              f"({r['seconds_per_run']:.0f}s per run)")

    # 3. facts for "explain what the sweep shows" 
    best_grid = best_by(runs, "grid")
    best_mode = best_by(runs, "mode_share")
    most_ur = max(runs, key=lambda r: r["summary"]["upper_right"]["mean"])
    facts = {
        "s_with_lowest_grid_distance": best_grid["s"],
        "s_with_lowest_mode_share_distance": best_mode["s"],
        "s_with_most_upper_right_samples": most_ur["s"],
        "exact_upper_right_share": target.upper_right_share,
    }
    print("3. Facts for your explanation (you write the explanation):")
    for k, v in facts.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    results = {"target": target.summary(),
               "noise_floor": floor, "grad_through_denoiser": True,
               "sweep": runs, "facts": facts,
               "seconds": time.time() - t_start}
    save_choice("guidance_s", best_grid["s"])  # the one number Part C needs
    write_sweep_markdown(results)
    sweep_figures(target, floor, runs, best_grid, most_ur)
    print(f"Done in {results['seconds']:.0f}s. See outputs/RESULTS.md.")


def write_sweep_markdown(r):
    fl = r["noise_floor"]
    lines = [
        "# Part B results: guided sampling with rules A and B", "",
        f"Target: {r['target']['name']}. Every row uses {N_EVAL:,} samples per seed, "
        f"seeds {SAMPLE_SEEDS}; ± is one standard deviation across seeds.", "",
        f"Sampler: DDPM, {T_STEPS} steps, gradient through the denoiser.", "",
        f"Exact share of the target in the upper-right quadrant: {r['target']['upper_right_share']:.4f}", "",
        "| Guidance scale s | Grid distance | Mode-share distance | Upper-right share | Non-finite samples |",
        "|---|---|---|---|---|",
    ]
    for run in r["sweep"]:
        m = run["summary"]
        lines.append(f"| {run['s']:g} | {fmt(m['grid'])} | {fmt(m['mode_share'])} | "
                     f"{fmt(m['upper_right'], 3)} | {m['non_finite']['mean']:.0f} |")
    lines.append(f"| exact samples (noise floor, {len(FLOOR_SEEDS)} seeds) | {fmt(fl['grid'])} | "
                 f"{fmt(fl['mode_share'])} | {fmt(fl['upper_right'], 3)} | 0 |")
    lines += ["", "## Mode shares (mean over seeds) beside the exact shares", "",
              "| Mode (angle) | Exact | " + " | ".join(f"s = {run['s']:g}" for run in r["sweep"]) + " |",
              "|---|---|" + "---|" * len(r["sweep"])]
    for k in range(ground_truth.N_MODES):
        row = f"| {k} ({ground_truth.ANGLES_DEG[k]:.1f}°) | {r['target']['mode_shares'][k]:.4f} | "
        row += " | ".join(f"{run['summary']['shares_mean'][k]:.4f}" for run in r["sweep"])
        lines.append(row + " |")
    lines += ["", "## Facts for the explanation", ""]
    lines += [f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}"
              for k, v in r["facts"].items()]
    write_section("part_b", lines, ["fig_b1_sweep.png", "fig_b2_samples.png",
                                    "fig_b3_mode_shares.png"])


def sweep_figures(target, floor, runs, best_grid, most_ur):
    s = np.array([r["s"] for r in runs])

    # Figure B1: the three curves against s
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    panels = [("grid", "Grid distance to the exact target"),
              ("mode_share", "Mode-share distance to the exact target"),
              ("upper_right", "Share of samples in the upper-right quadrant")]
    for ax, (key, title) in zip(axes, panels):
        mean = np.array([r["summary"][key]["mean"] for r in runs])
        sd = np.array([r["summary"][key]["sd"] for r in runs])
        for j in range(len(SAMPLE_SEEDS)):  # each seed as a faint dot
            ax.plot(s, [r["summary"][key]["values"][j] for r in runs], "o",
                    color="C0", alpha=0.25, ms=4)
        ax.errorbar(s, mean, yerr=sd, marker="o", capsize=3, color="C0",
                    label=f"guided model (mean of {len(SAMPLE_SEEDS)} seeds)")
        if key == "upper_right":
            ax.axhline(target.upper_right_share, color="C3", ls="--",
                       label="exact target's share")
        else:
            floor_band(ax, floor[key])
        ax.set_xlabel("guidance scale s  (0 = no guidance)")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("distance (0 = perfect, 1 = no overlap)")
    axes[2].set_ylabel("fraction of samples")
    fig.suptitle("Guided sampling with rules A and B: effect of the guidance scale", fontsize=13)
    fig.savefig(OUTPUTS / "fig_b1_sweep.png", dpi=150)
    plt.close(fig)

    # Figure B2: what the samples look like
    show = [runs[0], best_grid, most_ur, runs[-1]]
    labels = ["no guidance (s = 0)", f"lowest grid distance (s = {best_grid['s']:g})",
              f"most in upper-right (s = {most_ur['s']:g})", f"strongest (s = {runs[-1]['s']:g})"]
    imgs = [("exact target", target.grid_masses)] + [
        (lab, histogram_image(r["samples_first_seed"])) for lab, r in zip(labels, show)]
    vmax = max(im.max() for _, im in imgs)
    fig, axes = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 4.2), constrained_layout=True)
    for ax, (title, im) in zip(axes, imgs):
        h = ax.imshow(im.T, origin="lower", extent=[-3, 3, -3, 3], cmap="viridis", vmin=0, vmax=vmax)
        ax.axhline(0, color="w", lw=0.5, alpha=0.6)
        ax.axvline(0, color="w", lw=0.5, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("first coordinate")
    axes[0].set_ylabel("second coordinate")
    fig.colorbar(h, ax=axes, shrink=0.8, label="probability per grid cell")
    fig.suptitle("Exact target and guided samples (first seed)", fontsize=13)
    fig.savefig(OUTPUTS / "fig_b2_samples.png", dpi=150)
    plt.close(fig)

    # Figure B3: mode shares, exact vs two chosen scales
    k = np.arange(ground_truth.N_MODES)
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    ax.bar(k - w, target.mode_shares, w, label="exact target", color="gray")
    for off, r, lab, c in [(0, best_grid, f"s = {best_grid['s']:g} (lowest grid distance)", "C0"),
                           (w, most_ur, f"s = {most_ur['s']:g} (most in upper-right)", "C1")]:
        ax.bar(k + off, r["summary"]["shares_mean"], w, yerr=r["summary"]["shares_sd"],
               capsize=2, label=lab, color=c)
    ax.set_xticks(k, [f"mode {i}\n{ground_truth.ANGLES_DEG[i]:.1f}°" for i in k])
    ax.set_ylabel("share of samples nearest this mode")
    ax.set_title("Which modes the samples land in, against the exact target")
    ax.legend()
    fig.savefig(OUTPUTS / "fig_b3_mode_shares.png", dpi=150)
    plt.close(fig)


# Command 2: the design-choice test 
def design(s_values):
    make_dirs()
    model, sched = load_base()
    target = Target(RULES)
    floor = noise_floor(target, FLOOR_SEEDS, N_EVAL)

    variants = []
    for label, through in DESIGN_VARIANTS:  # only this one thing changes
        print(f"Variant: {label}")
        rows = []
        for s in s_values:
            r = run_setting(model, sched, target, RULES, s, grad_through_denoiser=through)
            m = r["summary"]
            print(f"   s = {s:>4}: grid {fmt(m['grid'])}  mode share {fmt(m['mode_share'])}  "
                  f"upper-right {fmt(m['upper_right'], 3)}  ({r['seconds_per_run']:.0f}s per run)")
            rows.append(r)
        variants.append({"label": label, "grad_through_denoiser": through, "runs": rows})

    lines = ["# Part B design choice: gradient through the denoiser or not", "",
             "Hypothesis: see HYPOTHESIS_B.md (committed before this run).", "",
             "| Variant | s | Grid distance | Mode-share distance | Upper-right share | Seconds per run |",
             "|---|---|---|---|---|---|"]
    for v in variants:
        for r in v["runs"]:
            m = r["summary"]
            lines.append(f"| {v['label']} | {r['s']:g} | {fmt(m['grid'])} | {fmt(m['mode_share'])} | "
                         f"{fmt(m['upper_right'], 3)} | {r['seconds_per_run']:.0f} |")
    lines.append(f"| exact samples (noise floor) | – | {fmt(floor['grid'])} | "
                 f"{fmt(floor['mode_share'])} | {target.upper_right_share:.3f} (exact) | – |")
    write_section("part_b_design", lines, ["fig_b4_design.png"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, key, title in [(axes[0], "grid", "Grid distance to the exact target"),
                           (axes[1], "mode_share", "Mode-share distance to the exact target")]:
        for i, v in enumerate(variants):
            ax.errorbar([r["s"] for r in v["runs"]],
                        [r["summary"][key]["mean"] for r in v["runs"]],
                        yerr=[r["summary"][key]["sd"] for r in v["runs"]],
                        marker="o", capsize=3, label=v["label"], color=f"C{i}")
        floor_band(ax, floor[key])
        ax.set_xlabel("guidance scale s")
        ax.set_ylabel("distance (0 = perfect)")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle(f"Design choice: does the gradient pass through the denoiser? "
                 f"({len(SAMPLE_SEEDS)} seeds each)", fontsize=12)
    fig.savefig(OUTPUTS / "fig_b4_design.png", dpi=150)
    plt.close(fig)
    print("Wrote outputs/part_b_design.md and outputs/fig_b4_design.png (see outputs/RESULTS.md)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("sweep", help="exact target, noise floor, and the sweep over s")
    d = sub.add_parser("design", help="gradient through the denoiser vs skipping it")
    d.add_argument("--s", type=float, nargs="+", default=DESIGN_S_VALUES,
                   help=f"scales to compare at (default {DESIGN_S_VALUES})")
    args = ap.parse_args()
    if args.command == "sweep":
        sweep()
    else:
        design(args.s)
