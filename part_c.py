"""
Part C: after a rule changes, when is editing the guide enough?

Rule A can live in two places and change in two ways: four cells.

    Where rule A lives            Before the change             Edit applied (no retraining)
    ----------------------------  ----------------------------  ------------------------------
    "guide": in a guide           full base model (all of p0)   swap guide A for guide A'
                                  + guides A and B              -> guides A' and B
    "baked": baked into the base  base model trained only on    add a guide for A'
             model                data with x1 > 0, + guide B   -> guides A' and B

    Change      New rule A'      Soft score
    tighten     x1 > 1.3         sigmoid((x1 - 1.3) / 0.5)
    reverse     x1 < 0           sigmoid(-x1 / 0.5)

In every cell the new exact target is  p0 * V_A'(x) * V_B(x)  (normalised).
Note that in the "baked" cells the edited sampler is (baked model) + guides A'
and B, which in effect also still contains the old rule A, through the data
the model was trained on. That is what makes this question interesting.

Commands:

  python part_c.py train-baked   train the baked-in model with the Part A code,
                                 training data filtered to x1 > 0
  python part_c.py cells         fill the four cells (trains the baked model first
                                 if its checkpoint is missing)
  python part_c.py all           both in order
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import diffusion_model as D
from src import ground_truth
from src.config import (BASE_BAKED_CKPT, BASE_FULL_CKPT, DATA_SEED,
                        FLOOR_SEEDS, N_EVAL, N_TRAIN_DATA, OUTPUTS, SAMPLE_SEEDS,
                        T_STEPS, TRAIN_SEED, TRAIN_STEPS, load_choice, make_dirs, write_section)
from src.scoring import fmt, histogram_image, noise_floor, run_setting
from src.rules import RULE_A_REVERSE, RULE_A_TIGHTEN, RULE_B
from src.ground_truth import Target

# ---------------------------------------------------------------------------
# The four cells
# ---------------------------------------------------------------------------
PLACES = {
    "guide": {"label": "rule A in a guide", "ckpt": BASE_FULL_CKPT},
    "baked": {"label": "rule A baked into the base model", "ckpt": BASE_BAKED_CKPT},
}
CHANGES = {"tighten": RULE_A_TIGHTEN, "reverse": RULE_A_REVERSE}


def training_data():
    """The same fixed training set as Part A (same seed)."""
    return ground_truth.sample_p0(N_TRAIN_DATA, np.random.default_rng(DATA_SEED))


def load(ckpt):
    model, info = D.load_model(ckpt)
    return model, D.make_schedule(info.get("T", T_STEPS)), info


def choose_s(s_arg):
    if s_arg is not None:
        return float(s_arg)
    s = load_choice("guidance_s")  # saved by `python part_b.py sweep`
    print(f"Using s = {s} (lowest grid distance in Part B).")
    return float(s)


# ---------------------------------------------------------------------------
# Command: train the baked-in model
# ---------------------------------------------------------------------------
def train_baked():
    """Part A code, training data filtered to x1 > 0. Same seeds and steps."""
    make_dirs()
    data = training_data()
    kept = data[data[:, 0] > 0]
    print(f"Training the baked-in model on {len(kept):,} of {len(data):,} points (x1 > 0 only), "
          f"{TRAIN_STEPS} steps")
    sched = D.make_schedule(T_STEPS)
    t0 = time.time()
    model, log = D.train(kept, sched, steps=TRAIN_STEPS, seed=TRAIN_SEED)
    secs = time.time() - t0
    D.save_model(model, BASE_BAKED_CKPT, T=T_STEPS, train_steps=TRAIN_STEPS,
                 data_seed=DATA_SEED, train_seed=TRAIN_SEED, seconds=secs,
                 training_data=f"p0 filtered to x1 > 0 ({len(kept)} points)",
                 loss_log=log)
    print(f"Saved {BASE_BAKED_CKPT} ({secs:.0f}s, final loss {log[-1][1]:.4f})")


# Command: the four cells
def cells(s):
    make_dirs()
    if not BASE_FULL_CKPT.exists():
        raise SystemExit(f"{BASE_FULL_CKPT} not found. Run `python part_a.py` first.")
    if not BASE_BAKED_CKPT.exists():
        train_baked()

    targets, floors = {}, {}
    for change, rule in CHANGES.items():
        targets[change] = Target([rule, RULE_B], name=f"p0 reweighted by {rule.name} and {RULE_B.name}")
        floors[change] = noise_floor(targets[change], FLOOR_SEEDS, N_EVAL)
        print(f"Target after '{change}': upper-right share {targets[change].upper_right_share:.3f}, "
              f"noise floor grid {fmt(floors[change]['grid'])}, "
              f"mode share {fmt(floors[change]['mode_share'])}")

    table, samples = {}, {}
    for place, P in PLACES.items():
        model, sched, _ = load(P["ckpt"])
        for change, rule in CHANGES.items():
            rules_after = [rule, RULE_B]  # the edit: A' and B as guides
            r = run_setting(model, sched, targets[change], rules_after, s,
                            keep_samples=True)
            m = r["summary"]
            excess = m["grid"]["mean"] - floors[change]["grid"]["mean"]
            print(f"Cell [{place} / {change}]: grid {fmt(m['grid'])} (above floor by {excess:.4f}), "
                  f"mode share {fmt(m['mode_share'])}, non-finite {m['non_finite']['mean']:.0f}")
            samples[(place, change)] = r["samples_first_seed"]
            table[f"{place}/{change}"] = {**r, "place": place, "change": change,
                                          "rules_after": [x.name for x in rules_after],
                                          "grid_excess_over_floor": excess}

    worst = max(table, key=lambda k: table[k]["grid_excess_over_floor"])
    print(f"Worst cell by grid distance above its noise floor: {worst}")
    out = {"s": s, "grad_through_denoiser": True, "targets": {c: t.summary() for c, t in targets.items()},
           "noise_floors": floors, "cells": table, "worst_cell": worst}
    write_cells_markdown(out)
    cells_figures(targets, samples, table, floors)
    print("Wrote outputs/part_c.md and figures fig_c1-c2 (see outputs/RESULTS.md).")


def write_cells_markdown(r):
    c = r["cells"]
    lines = ["# Part C results: four cells after a rule change", "",
             f"Guidance scale s = {r['s']:g} in every cell. {N_EVAL:,} samples per seed, "
             f"seeds {SAMPLE_SEEDS}; ± is one standard deviation across seeds.", "",
             "Each cell is scored against its NEW exact target: p0 reweighted by A' and B.", ""]
    for key, name in [("grid", "Grid distance"), ("mode_share", "Mode-share distance")]:
        lines += [f"## {name}", "",
                  "| Where rule A lives | Tighten (x1 > 1.3) | Reverse (x1 < 0) |", "|---|---|---|"]
        for place, P in PLACES.items():
            lines.append(f"| {P['label']} | {fmt(c[place + '/tighten']['summary'][key])} | "
                         f"{fmt(c[place + '/reverse']['summary'][key])} |")
        lines.append(f"| exact samples (noise floor) | {fmt(r['noise_floors']['tighten'][key])} | "
                     f"{fmt(r['noise_floors']['reverse'][key])} |")
        lines.append("")
    for change in CHANGES:
        exact = r["targets"][change]["mode_shares"]
        lines += [f"## Mode shares after '{change}' (mean ± sd over seeds)", "",
                  "| Mode (angle) | Exact | Guide cell | Baked cell |", "|---|---|---|---|"]
        for k in range(ground_truth.N_MODES):
            g, b = c[f"guide/{change}"]["summary"], c[f"baked/{change}"]["summary"]
            lines.append(f"| {k} ({ground_truth.ANGLES_DEG[k]:.1f}°) | {exact[k]:.4f} | "
                         f"{g['shares_mean'][k]:.4f} ± {g['shares_sd'][k]:.4f} | "
                         f"{b['shares_mean'][k]:.4f} ± {b['shares_sd'][k]:.4f} |")
        lines.append("")
    lines.append(f"Worst cell (grid distance minus its noise floor): **{r['worst_cell']}**")
    write_section("part_c", lines, ["fig_c1_cells.png", "fig_c2_mode_shares.png"])


def cells_figures(targets, samples, table, floors):
    # Figure C1: exact target and both cells, for each change
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.4), constrained_layout=True)
    for row, change in enumerate(CHANGES):
        imgs = [(f"exact target after '{change}'", targets[change].grid_masses)]
        for place, P in PLACES.items():
            g = table[f"{place}/{change}"]["summary"]["grid"]["mean"]
            imgs.append((f"{P['label']}\n(grid distance {g:.3f})",
                         histogram_image(samples[(place, change)])))
        vmax = max(im.max() for _, im in imgs)
        for col, (title, im) in enumerate(imgs):
            ax = axes[row, col]
            h = ax.imshow(im.T, origin="lower", extent=[-3, 3, -3, 3], cmap="viridis",
                          vmin=0, vmax=vmax)
            ax.axhline(0, color="w", lw=0.5, alpha=0.6)
            ax.axvline(CHANGES[change].threshold, color="w", lw=0.5, alpha=0.6)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("first coordinate")
            ax.set_ylabel("second coordinate")
        fig.colorbar(h, ax=axes[row, :], shrink=0.85, label="probability per grid cell")
    fig.suptitle("After the rule change: exact target and the two edited samplers (first seed)",
                 fontsize=13)
    fig.savefig(OUTPUTS / "fig_c1_cells.png", dpi=150)
    plt.close(fig)

    # Figure C2: mode shares per change
    k = np.arange(ground_truth.N_MODES)
    w = 0.27
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), constrained_layout=True)
    for ax, change in zip(axes, CHANGES):
        ax.bar(k - w, targets[change].mode_shares, w, color="gray", label="exact target")
        for off, place, col in [(0, "guide", "C0"), (w, "baked", "C1")]:
            sm = table[f"{place}/{change}"]["summary"]
            ax.bar(k + off, sm["shares_mean"], w, yerr=sm["shares_sd"], capsize=2,
                   color=col, label=PLACES[place]["label"])
        ax.set_xticks(k, [f"{i}\n{ground_truth.ANGLES_DEG[i]:.0f}°" for i in k])
        ax.set_xlabel("mode (angle of its centre)")
        ax.set_ylabel("share of samples nearest this mode")
        ax.set_title(f"After '{change}'")
        ax.legend(fontsize=9)
    fig.suptitle("Mode shares after the change, against the exact target", fontsize=13)
    fig.savefig(OUTPUTS / "fig_c2_mode_shares.png", dpi=150)
    plt.close(fig)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train-baked", "cells","all"])
    ap.add_argument("--s", type=float, default=None, help="guidance scale (default: Part B's lowest-grid-distance s)")

    args = ap.parse_args()

    if args.command == "train-baked":
        train_baked()
    elif args.command == "cells":
        cells(choose_s(args.s))
    
    else:
        s = choose_s(args.s)
        if not BASE_BAKED_CKPT.exists():
            train_baked()
        cells(s)

