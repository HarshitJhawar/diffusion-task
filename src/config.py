"""
Settings and seeds shared by every part. Change them here and only here,
so Parts A, B and C always use the same numbers.
"""

import json
from pathlib import Path

# ---- evaluation (fixed by the task sheet) ---------------------------------
N_EVAL = 20_000                       # samples per evaluation

# ---- base model -------------------------------------------------------------
N_TRAIN_DATA = 100_000                # size of the fixed training set drawn from p0
DATA_SEED = 0                         # seed for that training set
TRAIN_SEED = 0                        # seed for network initialisation and batches
T_STEPS = 1000                        # diffusion steps
TRAIN_STEPS = 20_000                  # Adam steps for a base model trained from scratch

# ---- seeds for sampling and for exact samples ------------------------------
SAMPLE_SEEDS = [1, 2, 3]              # model sampling: three seeds everywhere
FLOOR_SEEDS = list(range(100, 120))   # exact samples for noise floors (20 seeds)

# ---- folders ---------------------------------------------------------------
OUTPUTS = Path("outputs")        # every table, figure and saved choice lands here
CHECKPOINTS = Path("checkpoints")
BASE_FULL_CKPT = CHECKPOINTS / "base_full.pt"     # Part A: trained on all of p0
BASE_BAKED_CKPT = CHECKPOINTS / "base_baked.pt"   # Part C: trained only on x1 > 0


def make_dirs():
    for d in (OUTPUTS, CHECKPOINTS):
        d.mkdir(exist_ok=True)


# ---- values one step hands to a later step ----------------------------------
# One values need to survive between runs, so only they are saved:
#   "guidance_s"  Part B's best scale (lowest grid distance), used by Part C
CHOICES = OUTPUTS / "choices.json"


def save_choice(key, value):
    """Store one value in outputs/choices.json, keeping the others."""
    choices = json.loads(CHOICES.read_text()) if CHOICES.exists() else {}
    choices[key] = value
    OUTPUTS.mkdir(exist_ok=True)
    CHOICES.write_text(json.dumps(choices, indent=2))


def load_choice(key):
    """Read one value from outputs/choices.json, or stop with a clear message."""
    choices = json.loads(CHOICES.read_text()) if CHOICES.exists() else {}
    if key not in choices:
        raise SystemExit(f"'{key}' not found in {CHOICES}. Run the step that saves it first, "
                         "or pass it on the command line.")
    return choices[key]


# ---- all tables and figures in one file ---------------------------------------
def write_section(name, lines, figures=()):
    """Save one part's tables, with its figures shown below them, as
    outputs/<name>.md. Then rebuild outputs/RESULTS.md, which puts every part's
    section into one file (Part A first, then B, then C)."""
    text = "\n".join(list(lines) + [""] + [f"![{f}]({f})" for f in figures]) + "\n"
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / f"{name}.md").write_text(text, encoding="utf-8")
    sections = sorted(OUTPUTS.glob("part_*.md"))
    combined = "\n\n---\n\n".join(p.read_text(encoding="utf-8") for p in sections)
    (OUTPUTS / "RESULTS.md").write_text(combined, encoding="utf-8")
