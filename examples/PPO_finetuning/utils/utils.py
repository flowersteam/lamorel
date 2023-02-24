'''
PPO implementation taken from https://github.com/openai/spinningup
'''

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def scores_to_proba_dists(scores):
    proba_dists = []
    scores_max = torch.max(scores, dim=1)[0]
    for j in range(len(scores)):
        proba_dists.append(F.softmax(scores[j] / scores_max[j], dim=-1).unsqueeze(dim=0))

    proba_dists = torch.cat(proba_dists)
    dists = Categorical(probs=proba_dists)

    return dists