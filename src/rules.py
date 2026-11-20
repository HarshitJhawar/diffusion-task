"""
Every rule in the task has the same shape:
    V(x) = sigmoid( direction * (x[axis] - threshold) / temperature )

  Rule A          x1 > 0     -> axis 0, threshold 0,   direction +1
  Rule B          x2 > 0     -> axis 1, threshold 0,   direction +1
  Rule A' tight   x1 > 1.3   -> axis 0, threshold 1.3, direction +1
  Rule A' reverse x1 < 0     -> axis 0, threshold 0,   direction -1
  temperature 0.5 for all of them.

Guidance needs log V and its gradient. We compute log V with logsigmoid,
which stays accurate far on the wrong side, where sigmoid underflows to 0 and log(0) would be -inf.
"""

from dataclasses import dataclass 

import numpy as np 
import torch 
import torch.nn.functional as F 
from scipy.special import expit, log_expit # numpy sigmoid and log-sigmoid


@dataclass(frozen=True)
class Rule: 
    name: str
    axis: int 
    threshold: float 
    direction: int 

    def _z(self,x): 
        return self.direction * (x[...,self.axis] - self.threshold) / 0.5 

    # for exact targets and rejection sampling
    def score(self,x): 
        return expit(self._z(np.asarray(x,dtype=np.float64))) # sigmoid value. 

    def log_score(self,x): 
         return F.logsigmoid(self._z(x))

RULE_A = Rule("A: x1 > 0", axis=0, threshold=0.0, direction=+1)
RULE_B = Rule("B: x2 > 0", axis=1, threshold=0.0, direction=+1)
RULE_A_TIGHTEN = Rule("A' tighten: x1 > 1.3", axis=0, threshold=1.3, direction=+1)
RULE_A_REVERSE = Rule("A' reverse: x1 < 0", axis=0, threshold=0.0, direction=-1)

def combined_score(rules,x): 
    """Product of rule scores"""
    out = np.ones(np.asarray(x).shape[:-1])
    for r in rules: 
        out = out * r.score(x)
    return out 

def combined_log_score(rules,x): 
    """Sum of all log scores"""
    total = torch.zeros(x.shape[0],dtype=x.dtype)
    for r in rules: 
        total = total + r.log_score(x) 

    return total 
    
    