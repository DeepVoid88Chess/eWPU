"""Result combination utilities."""
import numpy as np

def combine_elementwise(parts):
    if not parts:
        return np.array([])
    return np.concatenate([np.asarray(p) for p in parts])

def combine_matmul(parts):
    if not parts:
        return np.empty((0, 0))
    return np.vstack([np.asarray(p) for p in parts])
