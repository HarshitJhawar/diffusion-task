"""
We take 20,000 samples generated the trained diffusion model (with guidance in part B and C), work out what fraction of samples falls in each box. Compute the total variation between those fractions and exact probabilities of the same boxes under the target distribution. 
The two total variation (TV) distances.
1. Grid distance       : TV between the sample histogram on the 60x60 grid
                         (plus one "outside the grid" bin) and the exact cell probabilities.
2. Mode-share distance : TV between the 8 shares of samples nearest each
                         centre and the exact 8 shares.

The exact shares of target distribution is generated in the ground_truth.py                          
"""
import time
import numpy as np
from .config import N_EVAL, SAMPLE_SEEDS
from .ground_truth import EDGES, N_MODES, nearest_mode
from .guidance import guided_sample

def total_variation(p,q): 
    p, q = np.asarray(p,float).ravel(), np.asarray(q,float).ravel() 
    assert p.shape == q.shape
    return 0.5 * np.abs(p-q).sum() 

def grid_histogram(samples): 
    """Returns (fractions of shape (60, 60), outside fraction). Together they sum to 1."""
    samples = np.asarray(samples) 
    counts,_,_ = np.histogram2d(samples[:,0], samples[:,1],bins=[EDGES,EDGES])
    n = len(samples) 
    return counts/n , 1.0- counts.sum()/n 

def grid_distance(samples,exact_masses,exact_outside):
    frac_grid,frac_out = grid_histogram(samples)
    p = np.append(frac_grid.ravel(),frac_out) 
    q = np.append(np.asarray(exact_masses).ravel(),exact_outside)
    return total_variation(p,q)  

def mode_shares(samples): 
    """ Share of samples whose nearest centre is mode k"""
    idx = nearest_mode(samples) 
    return np.bincount(idx,minlength=N_MODES) / len(idx) 

def mode_share_distance(samples,exact_shares): 
    return total_variation(mode_shares(samples),exact_shares) 


# Grading samples against a Target 
def score_samples(samples,target): 

    x = np.asarray(samples,dtype=np.float64) 
    bad = ~np.isfinite(x).all(axis=1) 
    x = x.copy()
    x[bad] = 1e6 # Move the bad samples outside the grid, counted in the "outside" cell 
    # This returns the 2 total variation distances above in a dict  
    return {
        "grid": grid_distance(x, target.grid_masses, target.grid_outside),
        "mode_share": mode_share_distance(x, target.mode_shares),
        "upper_right": float(((x[:, 0] > 0) & (x[:, 1] > 0)).mean()),
        "shares": mode_shares(x).tolist(),
        "non_finite": int(bad.sum()),
    }

def mean_sd(values):
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "values": [float(a) for a in v]}

def summarise(evals):
    """We take the score_samples output as input and calculate the mean_sd for nice summary for each of the seeds.
    """
    out = {k: mean_sd([e[k] for e in evals])
           for k in ("grid", "mode_share", "upper_right", "non_finite")}
    shares = np.array([e["shares"] for e in evals])
    out["shares_mean"] = shares.mean(0).tolist()
    out["shares_sd"] = (shares.std(0, ddof=1) if len(evals) > 1 else 0 * shares[0]).tolist()
    return out

def noise_floor(target,seeds,n): 
    return summarise([score_samples(target.rejection_sample(n, np.random.default_rng(s)), target) for s in seeds])

def fmt(d,digits=4): 
    """ used too print mean+- sd as text"""
    return f"{d['mean']:.{digits}f} ± {d['sd']:.{digits}f}"


def run_setting(model,sched,target,rules,s,seeds=SAMPLE_SEEDS,n=N_EVAL,keep_samples=False,grad_through_denoiser=True): 
    """Guided sampling with `rules` at scale `s`, one run per seed, each
    graded against `target`.

    Returns the summary over seeds (mean ± sd of each number), the per-seed scores, seconds per run, and (optionally) the first seed's samples for plots.
    """    
    evals, first = [], None 
    t0 = time.time() 

    for seed in seeds: 
        x = guided_sample(model,sched,n,seed,rules=rules,s=s,grad_through_denoiser=grad_through_denoiser)
        evals.append(score_samples(x,target))

        if keep_samples and first is None: 
            first = x 

    out = {"s": float(s), "grad_through_denoiser": grad_through_denoiser,
           "summary": summarise(evals), "per_seed": evals,
           "seconds_per_run": (time.time() - t0) / len(seeds)}
    if keep_samples:
        out["samples_first_seed"] = first
    return out

def best_by(results, key):
    """The result with the smallest mean of `key` (e.g. 'grid')."""
    return min(results, key=lambda r: r["summary"][key]["mean"])


def histogram_image(samples):
    """Fraction of (finite) samples per grid cell, for plotting (60 x 60)."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    return grid_histogram(x)[0]