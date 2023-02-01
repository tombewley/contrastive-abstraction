import numpy as np
from numba import jit


def data_bounds(*data):
    data = np.concatenate(data)
    if data.shape[0] == 0:
        return np.array([np.full(data.shape[1], -np.inf), np.full(data.shape[1], np.inf)])
    return np.array([data.min(axis=0), data.max(axis=0)])


@jit(nopython=True, cache=True, parallel=True)
def geq_threshold_mask(data: np.ndarray, init_mask: np.ndarray, dim: int, thresholds: np.ndarray):
    mask = np.zeros((thresholds.shape[0], data.shape[0]), dtype=np.bool_)
    mask[:, init_mask] = data[init_mask, dim].copy().reshape(1, -1) >= thresholds.reshape(-1, 1)
    return mask


def jsd(counts: np.ndarray):
    if len(counts.shape) == 3:
        counts = np.expand_dims(counts, 0)
    return _jsd(counts)


@jit(nopython=True, cache=True)
def _jsd(counts: np.ndarray):
    def entropy(p):
        return (-p * np.where(p > 0, np.log(p), 0)).sum(axis=-1)
    r, n, m, _ = counts.shape
    counts = counts.reshape(r, n, m*m)  # Flatten 2D representation of joint distribution
    counts_total = counts.sum(axis=-1)
    h_unmixed = entropy(counts / counts_total.reshape(r, n, 1))
    counts_mixed = counts.sum(axis=1)
    h_mixed = entropy(counts_mixed / counts_mixed.sum(axis=-1).reshape(r, 1))
    return h_mixed - (h_unmixed * counts_total / counts_total.sum(axis=-1).reshape(r, 1)).sum(axis=-1)


@jit(nopython=True, cache=True)
def softmax(x, tau):
    x = x / tau
    x = x - x.max()  # For stability
    return np.exp(x) / np.exp(x).sum()
