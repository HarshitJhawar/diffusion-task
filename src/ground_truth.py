import numpy as np 
from scipy.special import ndtr 
from .rules import combined_score

N_MODES = 8
RADIUS = 2.0 
STD = 0.15 
ANGLES_DEG = 22.5 + 45.0 * np.arange(N_MODES)
CENTRES = RADIUS * np.stack(
    [np.cos(np.deg2rad(ANGLES_DEG)), np.sin(np.deg2rad(ANGLES_DEG))], axis=1
) # centres of gaussian = (rcos_theta,r.sin_theta)

# Gaussian mixture: Sample from 8 Gaussian with equal probability 
def sample_p0(n,rng): # rng is a random number generator object 
    k = rng.integers(0,N_MODES,size=n) # Generate n integers from uniform discrete distribution [0,N_MODES] 
    return CENTRES[k] + STD * rng.standard_normal((n,2))

def density_p0(x): 
    """ Calculate exact density p0(x) at points x of shape (..., 2).
    p0(x) = (1/8) * sum_k N(x; centre_k, STD^2 I)
    """
    x = np.asarray(x,dtype=np.float64)

    d2 = ((x[..., None, :] - CENTRES) ** 2).sum(-1)          # (..., 8) squared distances
    comp = np.exp(-d2 / (2 * STD**2)) / (2 * np.pi * STD**2)  # density for each Gaussian 
    return comp.mean(-1)                                       # equal weights 1/8 


GRID_LO, GRID_HI, GRID_BINS = -3.0, 3.0, 60 # Grid with extremes (-3,-3) and (3,3)
EDGES = np.linspace(GRID_LO, GRID_HI, GRID_BINS + 1) 
# We keep one extra "outside cell" so that grid probabilities sum to exactly 1. 

def exact_grid_masses_p0(): 
    """
    Generate exact pmf of each grid cell under p0. We use an accurate isotropic Gaussian trick such that numerical integration is not needed. 
    [Phi((x_hi - mx)/s) - Phi((x_lo - mx)/s)] * [same thing for y].
    """
    masses = np.zeros((GRID_BINS,GRID_BINS))
    for mx,my in CENTRES: 
        px = np.diff(ndtr((EDGES - mx) / STD))  # (60,) prob of each x-interval
        py = np.diff(ndtr((EDGES - my) / STD))  # (60,) prob of each y-interval
        masses += np.outer(px,py) / N_MODES
    return masses, 1.0 - masses.sum() 

def numerical_grid_masses(unnorm_density, sub=20, box=5.0, box_step=0.01): 
    """ Grid-cell probabilities for density. 
    Used for probability distributions with closed form, e.g. p*(x) ∝ p0(x) V_A(x) V_B(x)
    Also used in the tests to cross-check exact_grid_masses_p0.
    """

    # Normalising constant Z = integral of the unnormalised density
    c = np.arange(-box + box_step / 2, box, box_step)
    X, Y = np.meshgrid(c, c, indexing="ij")
    Z = unnorm_density(np.stack([X, Y], -1)).sum() * box_step**2

    # Cell integrals on a sub-grid
    h = (GRID_HI - GRID_LO) / (GRID_BINS * sub)
    c = GRID_LO + h * (np.arange(GRID_BINS * sub) + 0.5)
    X, Y = np.meshgrid(c, c, indexing="ij")
    vals = unnorm_density(np.stack([X, Y], -1)) * h**2 / Z
    masses = vals.reshape(GRID_BINS, sub, GRID_BINS, sub).sum(axis=(1, 3))
    return masses, 1.0 - masses.sum()

def nearest_mode(x): 
    """"Returns the index of the newarest centre for point in of shape (n,2). """
    dist = ((np.asarray(x)[:, None, :] - CENTRES) ** 2).sum(-1)
    return dist.argmin(1)

def exact_mode_shares(unnorm_density,box=5.0, step=0.005): # 
    """ Exact probability that a random point is nearest to which centre. 
    Integrate the density over a fine grid, give each grid point to its nearest centre, and add up per centre. 
    We know for p0, it is 1/8 for each mode. 
    """

    c = np.arange(-box + step /2, box,step) 
    shares = np.zeros(N_MODES)
    for row in np.array_split(c,20): 
        X, Y = np.meshgrid(row,c,indexing='ij')
        pts = np.stack([X.ravel(),Y.ravel()],1) 
        w = unnorm_density(pts) 
        shares += np.bincount(nearest_mode(pts), weights=w,minlength=N_MODES)
    return shares / shares.sum() 

EXACT_MODE_SHARES_P0 = np.full(N_MODES, 1.0 / N_MODES)  # by symmetry


class Target:
    """Everything exact about  p*(x) ∝ p0(x) * V_1(x) * V_2(x) * ...

    We perform rejection sampling to give p*: point x is proposed with
    probability p0(x), kept with probability V(x), so kept points have
    density proportional to p0(x) V(x). The fraction of points kept is
    Z = E_{x ~ p0}[V(x)], the normalising constant.
    
    With no rules, Target([]) is plain p0.
    """

    def __init__(self,rules,name=None,box=5.0, step=0.005): 
        self.rules = list(rules)
        self.name = name or " and ".join(r.name for r in self.rules)

        self.grid_masses, self.grid_outside = numerical_grid_masses(self.unnorm_density)
        self.mode_shares = exact_mode_shares(self.unnorm_density)
        self.upper_right_share = float(self.grid_masses[30:, 30:].sum())

    def unnorm_density(self, x): 
        """p0(x) * V(x): the target up to a constant.
        This is the unnormalized density. 
        """
        return density_p0(x) * combined_score(self.rules, x)

    def rejection_sample(self, n, rng):
        """n exact samples: propose from p0, keep each with probability V(x)."""
        kept = []
        while sum(len(k) for k in kept) < n:
            x = sample_p0(4 * n, rng) # sample from p0 if 
            kept.append(x[rng.random(len(x)) < combined_score(self.rules, x)])
        return np.concatenate(kept)[:n]

    def summary(self): 
        return {"name": self.name, "mode_shares": self.mode_shares.tolist(),
                "upper_right_share": self.upper_right_share,
                "mass_outside_grid": float(self.grid_outside)}