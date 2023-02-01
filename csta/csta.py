import numpy as np
import numba


class ContrastiveAbstraction:
    """
    Class for performing contrastive abstraction of context-labelled state transition data.
    """
    def __init__(self, context_dims: list[str], context_split_thresholds: list[np.ndarray],
                 state_dims: list[str], state_split_thresholds: list[np.ndarray]):
        assert len(context_split_thresholds) == len(context_dims), "Must specify split thresholds for every dim."
        assert len(state_split_thresholds) == len(state_dims), "Must specify split thresholds for every dim."
        self.W = HRSubset(dims=context_dims)
        self.context_split_thresholds = context_split_thresholds
        self.X = HRSubset(dims=state_dims)
        self.state_split_thresholds = state_split_thresholds

    @property
    def n(self): return len(self.W.leaves)
    @property
    def m(self): return len(self.X.leaves)
    @property
    def d_c(self): return len(self.W.dims)
    @property
    def d_s(self): return len(self.X.dims)

    def transition_mask(self, contexts: np.ndarray, states: np.ndarray, next_states: np.ndarray):
        num = contexts.shape[0]
        assert states.shape[0] == next_states.shape[0] == num, "Arrays must have common length."
        l_W, l_X = self.W.leaves, self.X.leaves
        mask = np.zeros((len(l_W), len(l_X), len(l_X), num), dtype=np.bool_)
        def _recurse(w, x, xx, indices):
            w_l = w_r = w; x_l = x_r = x; xx_l = xx_r = xx
            if not w.is_leaf:
                right = w.sort(contexts[indices])
                w_l, w_r = w.left, w.right
            elif not x.is_leaf:
                right = x.sort(states[indices])
                x_l, x_r = x.left, x.right
            elif not xx.is_leaf:
                right = xx.sort(next_states[indices])
                xx_l, xx_r = xx.left, xx.right
            else:
                mask[l_W.index(w), l_X.index(x), l_X.index(xx), indices] = True
                return
            _recurse(w_l, x_l, xx_l, indices[~right])
            _recurse(w_r, x_r, xx_r, indices[ right])
        _recurse(self.W, self.X, self.X, np.arange(num))
        return mask

    def transition_counts(self, contexts: np.ndarray, states: np.ndarray, next_states: np.ndarray):
        return self.transition_mask(contexts, states, next_states).sum(axis=3)

    def set_1d_context_windows(self, *thresholds):
        self.W.merge()
        window = self.W
        for threshold in sorted(thresholds):
            window.split(0, threshold)
            window = window.right

    def eval_changes(self, contexts: np.ndarray, states: np.ndarray, next_states: np.ndarray,
                     context_split: bool = False, context_merge: bool = False,
                     state_split: bool = False, state_merge: bool = False,
                     debug: bool = False):
        n, m = self.n, self.m
        context_split_gains, context_merge_gains, state_split_gains, state_merge_gains = None, None, None, None
        # Propagate the dataset through the extant abstractions and compute JSD as a baseline
        mask_current = self.transition_mask(contexts, states, next_states)
        counts_current = mask_current.sum(axis=3)
        jsd_current = jsd(counts_current)[0]
        if context_split:
            assert m > 1, "Context splitting requires at least two abstract states."
            w_mask_current = mask_current.sum(axis=(1, 2)).astype(bool)
            context_split_gains = [[] for _ in range(n)]
            for w in range(n):
                bounds = data_bounds(contexts[w_mask_current[w]])
                # Define expanded mask and counts arrays
                mask_exp = np.insert(mask_current, w + 1, 0, axis=0)
                counts_exp = np.insert(counts_current, w + 1, 0, axis=0)
                for dim in range(self.d_c):
                    # Valid thresholds are those that lie within the bounds of the observations in this window
                    th = self.context_split_thresholds[dim]
                    valid_thresholds = th[np.logical_and(bounds[0, dim] < th, th < bounds[1, dim])]
                    # Compare all context values to all valid thresholds in parallel
                    move = geq_threshold_mask(contexts, w_mask_current[w], dim, valid_thresholds)
                    delta = np.zeros((len(valid_thresholds), n + 1, m, m), dtype=np.int_)
                    for c in range(len(valid_thresholds)):
                        # Handle transitions where the context is in the right child
                        delta_w = mask_exp[w, :, :, move[c]].sum(axis=0)
                        delta[c, w] -= delta_w
                        delta[c, w + 1] += delta_w
                        assert (delta[c].sum(axis=0) == 0).all()
                    # Record Jensen-Shannon divergence for each valid threshold
                    context_split_gains[w].append((valid_thresholds, _jsd(counts_exp + delta) - jsd_current))
                    # Check that computed counts match that after split is made
                    if debug and len(valid_thresholds) > 0:
                        greedy = np.argmax(context_split_gains[w][-1][1])
                        parent = self.W.leaves[w]
                        parent.split(dim, valid_thresholds[greedy])
                        match = (self.transition_counts(contexts, states, next_states)
                                 == (counts_exp + delta[greedy])).all()
                        assert match
                        parent.merge()
        if context_merge:
            assert n > 1, "Context merging requires at least two windows."
            context_merge_gains = []
            transition_sums = counts_current.sum(axis=0)
            for parent, ws in self.W.merge_sets:
                # Merge counts along window axis
                counts_merged = np.concatenate([counts_current[:ws[0]],
                                                counts_current[ws].sum(axis=0, keepdims=True),
                                                counts_current[ws[-1] + 1:]], axis=0)
                assert (counts_merged.sum(axis=0) == transition_sums).all()
                # Record Jensen-Shannon divergence for each merge set
                context_merge_gains.append((parent, ws, jsd(counts_merged)[0] - jsd_current))
        if state_split:
            assert n > 1, "State splitting requires at least two context windows."
            x_mask_current = mask_current.sum(axis=(0, 2)).astype(bool)
            xx_mask_current = mask_current.sum(axis=(0, 1)).astype(bool)
            state_split_gains = [[] for _ in range(m)]
            for x in range(m):
                bounds = data_bounds(states[x_mask_current[x]], next_states[xx_mask_current[x]])
                # Define expanded mask and counts arrays
                mask_exp = np.insert(mask_current, x + 1, 0, axis=1)
                mask_exp = np.insert(mask_exp, x + 1, 0, axis=2)
                counts_exp = np.insert(counts_current, x + 1, 0, axis=1)
                counts_exp = np.insert(counts_exp, x + 1, 0, axis=2)
                for dim in range(self.d_s):
                    # Valid thresholds are those that lie within the bounds of the observations in this abstract state
                    th = self.state_split_thresholds[dim]
                    valid_thresholds = th[np.logical_and(bounds[0, dim] < th, th < bounds[1, dim])]
                    # Compare all state values to all valid thresholds in parallel
                    move_x = geq_threshold_mask(states, x_mask_current[x], dim, valid_thresholds)
                    move_xx = geq_threshold_mask(next_states, xx_mask_current[x], dim, valid_thresholds)
                    move_both = np.logical_and(move_x, move_xx)
                    move_x_only = np.logical_and(move_x, ~move_xx)
                    move_xx_only = np.logical_and(~move_x, move_xx)
                    delta = np.zeros((len(valid_thresholds), n, m + 1, m + 1), dtype=np.int_)
                    for c in range(len(valid_thresholds)):
                        # Handle transitions where the first state is in the right child
                        delta_x_only = mask_exp[:, x, :, move_x_only[c]].sum(axis=0)
                        delta[c, :, x, :] -= delta_x_only
                        delta[c, :, x + 1, :] += delta_x_only
                        # Handle transitions where the second state is in the right child
                        delta_xx_only = mask_exp[:, :, x, move_xx_only[c]].sum(axis=2)
                        delta[c, :, :, x] -= delta_xx_only
                        delta[c, :, :, x + 1] += delta_xx_only
                        # Handle transitions where both states are in the right child
                        delta_both = mask_exp[:, x, x, move_both[c]].sum(axis=1)
                        delta[c, :, x, x] -= delta_both
                        delta[c, :, x + 1, x + 1] += delta_both
                        assert (delta[c].sum(axis=(1, 2)) == 0).all()
                    # Record Jensen-Shannon divergence for each valid threshold
                    state_split_gains[x].append((valid_thresholds, _jsd(counts_exp + delta) - jsd_current))
                    # Check that computed counts match that after split is made
                    if debug and len(valid_thresholds) > 0:
                        greedy = np.argmax(state_split_gains[x][-1][1])
                        parent = self.X.leaves[x]
                        parent.split(dim, valid_thresholds[greedy])
                        match = (self.transition_counts(contexts, states, next_states)
                                 == (counts_exp + delta[greedy])).all()
                        assert match
                        parent.merge()
        if state_merge:
            assert m > 1, "State merging requires at least two abstract states."
            state_merge_gains = []
            window_sums = counts_current.sum(axis=(1, 2))
            for parent, xs in self.X.merge_sets:
                # Merge counts along first state axis
                counts_merged = np.concatenate([counts_current[:, :xs[0]],
                                                counts_current[:, xs].sum(axis=1, keepdims=True),
                                                counts_current[:, xs[-1] + 1:]], axis=1)
                # Merge counts along second state axis
                counts_merged = np.concatenate([counts_merged[:, :, :xs[0]],
                                                counts_merged[:, :, xs].sum(axis=2, keepdims=True),
                                                counts_merged[:, :, xs[-1] + 1:]], axis=2)
                assert (counts_merged.sum(axis=(1, 2)) == window_sums).all()
                # Record Jensen-Shannon divergence for each merge set
                state_merge_gains.append((parent, xs, jsd(counts_merged)[0] - jsd_current))
        return context_split_gains, context_merge_gains, state_split_gains, state_merge_gains

    def make_greedy_change(self, contexts: np.ndarray, states: np.ndarray, next_states: np.ndarray,
                           alpha: float = 0., beta: float = 0., power: float = 1., tau: float = 0.):
        n, m = self.n, self.m
        context_split_gains, context_merge_gains, state_split_gains, state_merge_gains = self.eval_changes(
            contexts, states, next_states,
            context_split=(m > 1), state_split=(n > 1),
            context_merge=(m > 1 and n > 1), state_merge=(m > 1 and n > 1),
        )
        candidates, quals = [], []
        if m > 1:
            for w in range(self.n):
                for dim in range(self.d_c):
                    g = context_split_gains[w][dim]
                    if len(g[1]) > 0:
                        greedy = np.argmax(g[1])
                        qual = g[1][greedy] - beta * ((n + 1)**power - n**power)
                        if qual > 0:
                            candidates.append(("context split", w, dim, g[0][greedy]))
                            quals.append(qual)
        if n > 1:
            for x in range(self.m):
                for dim in range(self.d_s):
                    g = state_split_gains[x][dim]
                    if len(g[1]) > 0:
                        greedy = np.argmax(g[1])
                        qual = g[1][greedy] - alpha * ((m + 1)**power - m**power)
                        if qual > 0:
                            candidates.append(("state split", x, dim, g[0][greedy]))
                            quals.append(qual)
        if m > 1 and n > 1:
            for parent, ws, gain in context_merge_gains:
                qual = gain - beta * ((n + 1 - len(ws))**power - n**power)
                if qual > 0:
                    candidates.append(("context merge", parent, ws))
                    quals.append(qual)
            for parent, xs, gain in state_merge_gains:
                qual = gain - alpha * ((m + 1 - len(xs))**power - m**power)
                if qual > 0:
                    candidates.append(("state merge", parent, xs))
                    quals.append(qual)
        if len(candidates) == 0:
            print(f"(n={self.n}, m={self.m}) No change; at local optimum")
        else:
            if tau > 0:
                chosen = np.random.choice(np.arange(len(candidates)), p=softmax(np.array(quals), tau=tau))
            else:
                chosen = np.argmax(quals)
            if candidates[chosen][0] == "context split":
                _, w, dim, threshold = candidates[chosen]
                self.W.leaves[w].split(dim, threshold)
                print(f"(n={self.n}, m={self.m}) Split window {w} at {self.W.dims[dim]} = {threshold}")
            elif candidates[chosen][0] == "context merge":
                _, parent, ws = candidates[chosen]
                parent.merge()
                print(f"(n={self.n}, m={self.m}) Merge windows {ws}")
            elif candidates[chosen][0] == "state split":
                _, x, dim, threshold = candidates[chosen]
                self.X.leaves[x].split(dim, threshold)
                print(f"(n={self.n}, m={self.m}) Split abstract state {x} at {self.X.dims[dim]} = {threshold}")
            elif candidates[chosen][0] == "state merge":
                _, parent, xs = candidates[chosen]
                parent.merge()
                print(f"(n={self.n}, m={self.m}) Merge abstract states {xs}")
        return candidates


class HRSubset:
    def __init__(self, dims: list, bounds: np.ndarray = None):
        self.dims = dims
        if bounds is None:
            self.bounds = np.stack([np.full(len(self.dims), -np.inf), np.full(len(self.dims), np.inf)])
        else:
            assert bounds.shape[0] == 2, "Invalid bounds shape."
            self.bounds = bounds
        self.merge()

    def __repr__(self):
        return "(" + " and ".join([f"{l} <= {dim} < {u}" for dim, (l, u) in zip(self.dims, self.bounds.T)]) + ")"

    @property
    def is_leaf(self):
        return self.split_dim is None

    @property
    def leaves(self):
        if self.is_leaf:
            return [self]
        return self.left.leaves + self.right.leaves

    @property
    def merge_sets(self):
        leaves = self.leaves
        merge_sets_ = []
        def _recurse(subset):
            if subset.is_leaf: return
            merge_sets_.append((subset, np.array([leaves.index(l) for l in subset.leaves])))
            _recurse(subset.left); _recurse(subset.right)
        _recurse(self)
        return merge_sets_

    def sort(self, data: np.ndarray):
        assert not self.is_leaf, "Subset hasn't been split."
        return data[..., self.split_dim] >= self.split_threshold

    def merge(self):
        self.split_dim, self.split_threshold, self.left, self.right = None, None, None, None

    def split(self, dim: int, threshold: float):
        assert self.is_leaf, "Subset has already been split."
        assert 0 <= dim < len(self.dims), f"Split dimension ({dim}) out of range."
        assert self.bounds[0, dim] <= threshold < self.bounds[1, dim], f"Split (dim {dim} = {threshold}) out of bounds."
        self.split_dim = dim
        self.split_threshold = threshold
        bounds_left, bounds_right = self.bounds.copy(), self.bounds.copy()
        bounds_left[1, dim] = bounds_right[0, dim] = threshold
        self.left = HRSubset(self.dims, bounds_left)
        self.right = HRSubset(self.dims, bounds_right)


def data_bounds(*data):
    data = np.concatenate(data)
    if data.shape[0] == 0:
        return np.array([np.full(data.shape[1], -np.inf), np.full(data.shape[1], np.inf)])
    return np.array([data.min(axis=0), data.max(axis=0)])


@numba.jit(nopython=True, cache=True, parallel=True)
def geq_threshold_mask(data: np.ndarray, init_mask: np.ndarray, dim: int, thresholds: np.ndarray):
    mask = np.zeros((thresholds.shape[0], data.shape[0]), dtype=np.bool_)
    mask[:, init_mask] = data[init_mask, dim].copy().reshape(1, -1) >= thresholds.reshape(-1, 1)
    return mask


def jsd(counts: np.ndarray):
    if len(counts.shape) == 3:
        counts = np.expand_dims(counts, 0)
    return _jsd(counts)


@numba.jit(nopython=True, cache=True)
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


@numba.jit(nopython=True, cache=True)
def softmax(x, tau):
    x = x / tau
    x = x - x.max()  # For stability
    return np.exp(x) / np.exp(x).sum()
