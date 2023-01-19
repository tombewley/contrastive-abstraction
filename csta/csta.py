import numpy as np
import numba


class CSTA:
    """
    Class for performing contrastive spatiotemporal abstraction.
    """
    def __init__(self, temporal_dims: list[str], spatial_dims: list[str]):
        self.T = HRSubset(dims=temporal_dims)
        self.X = HRSubset(dims=spatial_dims)

    @property
    def n(self): return len(self.T.leaves)
    @property
    def m(self): return len(self.X.leaves)

    def transition_mask(self, timestamps: np.ndarray, states: np.ndarray, next_states: np.ndarray):
        return transition_mask(w_bounds_array=np.array([w.bounds for w in self.T.leaves]),
                               x_bounds_array=np.array([x.bounds for x in self.X.leaves]),
                               timestamps=timestamps, states=states, next_states=next_states
                               )

    # def set_windows(self, *thresholds):
    #     self.T.merge()
    #     for threshold in sorted(thresholds):
    #         self.T.leaves[-1].split(0, threshold)

    def eval_splits(self, timestamps: np.ndarray, states: np.ndarray, next_states: np.ndarray, temporal: bool = False):
        n, m = self.n, self.m
        mask_current = self.transition_mask(timestamps, states, next_states)
        counts_current = mask_current.sum(axis=3)
        jsd_current = jsd(counts_current)
        if temporal:
            assert m > 1, "Temporal abstraction requires at least two abstract states."
            w_mask_current = mask_current.sum(axis=(1, 2)).astype(bool)
            qual = []
            raise NotImplementedError
        else:
            assert n > 1, "Spatial abstraction requires at least two time windows."
            x_mask_current = mask_current.sum(axis=(0, 2)).astype(bool)
            xx_mask_current = mask_current.sum(axis=(0, 1)).astype(bool)
            qual = []
            for x in numba.prange(len(self.X.leaves)):
                qual.append([])
                mask_exp = np.insert(mask_current, x + 1, 0, axis=1)
                mask_exp = np.insert(mask_exp,     x + 1, 0, axis=2)
                counts_exp = np.insert(counts_current, x + 1, 0, axis=1)
                counts_exp = np.insert(counts_exp, x + 1, 0, axis=2)
                for dim in numba.prange((len(self.X.dims))):

                    thresholds = np.linspace(0.1, 9.9, 100)

                    move_x  = geq_threshold_mask(states,      x_mask_current[x],  dim, thresholds)
                    move_xx = geq_threshold_mask(next_states, xx_mask_current[x], dim, thresholds)
                    move_both = np.logical_and(move_x, move_xx)
                    move_x_only = np.logical_and(move_x, ~move_xx)
                    move_xx_only = np.logical_and(~move_x, move_xx)

                    delta = np.zeros((len(thresholds), n, m + 1, m + 1), dtype=np.int_)
                    for c in numba.prange(len(thresholds)):

                        delta_x_only = mask_exp[:, x, :, move_x_only[c]].sum(axis=0)
                        delta[c, :, x, :] -= delta_x_only
                        delta[c, :, x + 1, :] += delta_x_only

                        delta_xx_only = mask_exp[:, :, x, move_xx_only[c]].sum(axis=2)
                        delta[c, :, :, x] -= delta_xx_only
                        delta[c, :, :, x + 1] += delta_xx_only

                        delta_both = mask_exp[:, x, x, move_both[c]].sum(axis=1)
                        delta[c, :, x, x] -= delta_both
                        delta[c, :, x + 1, x + 1] += delta_both

                        assert (delta[c].sum(axis=(1, 2)) == 0).all()
                        # TODO: More assertions

                    qual[x].append(jsd(counts_exp + delta) - jsd_current)

        return qual

    def eval_merges(self):
        """MCCP"""


class HRSubset:
    def __init__(self, dims: list, bounds: np.ndarray = None):
        self.dims = dims
        if bounds is None:
            self.bounds = np.stack([np.full(len(self.dims), -np.inf), np.full(len(self.dims), np.inf)])
        else:
            assert bounds.shape[0] == 2, "Invalid bounds shape."
            self.bounds = bounds
        self.merge()

    def __str__(self):
        return ", ".join([f"{l} <= {dim} < {u}" for dim, (l, u) in zip(self.dims, self.bounds.T)])

    @property
    def leaves(self):
        if self.split_dim is None:
            return [self]
        return self.left.leaves + self.right.leaves

    def contains(self, data: np.ndarray):
        return in_bounds_mask(self.bounds, data)

    def merge(self):
        self.split_dim, self.split_threshold, self.left, self.right = None, None, None, None

    def split(self, dim: int, threshold: float):
        assert self.split_dim is None, "Subset has already been split."
        assert 0 <= dim < len(self.dims), f"Split dimension ({dim}) out of range."
        assert self.bounds[0, dim] <= threshold < self.bounds[1, dim], f"Split (dim {dim} = {threshold}) out of bounds."
        self.split_dim = dim
        self.split_threshold = threshold
        bounds_left, bounds_right = self.bounds.copy(), self.bounds.copy()
        bounds_left[1, dim] = bounds_right[0, dim] = threshold
        self.left = HRSubset(self.dims, bounds_left)
        self.right = HRSubset(self.dims, bounds_right)


@numba.jit(nopython=True, cache=True, parallel=True)
def in_bounds_mask(bounds, data):
    mask = np.zeros(data.shape[0], dtype=np.bool_)
    for i in numba.prange(data.shape[0]):
        if (bounds[0] <= data[i]).all() and (data[i] < bounds[1]).all():
            mask[i] = True
    return mask


@numba.jit(nopython=True, cache=True, parallel=True)
def transition_mask(w_bounds_array: np.ndarray, x_bounds_array: np.ndarray,
                     timestamps: np.ndarray, states: np.ndarray, next_states: np.ndarray):
    num = timestamps.shape[0]
    assert states.shape[0] == next_states.shape[0] == num, "Arrays must have common length."
    n, m = w_bounds_array.shape[0], x_bounds_array.shape[0]
    mask = np.zeros((n, m, m, num), dtype=np.bool_)
    idx_range = np.arange(num)
    for w in numba.prange(n):
        mask_w = in_bounds_mask(w_bounds_array[w], timestamps)
        for x in numba.prange(m):
            mask_x = in_bounds_mask(x_bounds_array[x], states[mask_w])
            for xx in numba.prange(m):
                idx_range_here = idx_range[mask_w][mask_x]
                mask_here = in_bounds_mask(x_bounds_array[xx], next_states[mask_w][mask_x])
                for i in numba.prange(len(idx_range_here)):
                    mask[w, x, xx, idx_range_here[i]] = mask_here[i]
    return mask


@numba.jit(nopython=True, cache=True, parallel=True)
def geq_threshold_mask(data: np.ndarray, init_mask: np.ndarray, dim: int, thresholds: np.ndarray):
    mask = np.zeros((thresholds.shape[0], data.shape[0]), dtype=np.bool_)
    mask[:, init_mask] = data[init_mask, dim].copy().reshape(1, -1) >= thresholds.reshape(-1, 1)
    return mask


def jsd(counts: np.ndarray):
    if len(counts.shape) == 3:
        counts = np.expand_dims(counts, 0)
    return _jsd(counts)


# @numba.jit(nopython=True, cache=True)
def _jsd(counts: np.ndarray):
    def entropy(p):  # https://stackoverflow.com/a/65675506
        plogp = (-p * np.log(p)).ravel()
        plogp[np.isnan(plogp)] = 0  # -p*log(p) undefined at 0
        return plogp.reshape(p.shape).sum(axis=-1)
    r, n, m, _ = counts.shape
    counts = counts.reshape(r, n, m*m)  # Flatten 2D representation of joint distribution
    counts_total = counts.sum(axis=-1)
    h_unmixed = entropy(counts / counts_total.reshape(r, n, 1))
    counts_mixed = counts.sum(axis=1)
    h_mixed = entropy(counts_mixed / counts_mixed.sum(axis=-1).reshape(r, 1))
    return h_mixed - (h_unmixed * counts_total / counts_total.sum(axis=-1).reshape(r, 1)).sum(axis=-1)
