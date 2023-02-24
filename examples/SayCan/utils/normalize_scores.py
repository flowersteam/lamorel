'''
  Code taken from https://github.com/google-research/google-research/blob/master/saycan/SayCan-Robot-Pick-Place.ipynb
  Licensed under Apache 2.0 license
'''

import numpy as np

def normalize_scores(scores):
    max_score = max(scores.values())
    normed_scores = {key: np.clip(scores[key] / max_score, 0, 1) for key in scores}
    return normed_scores