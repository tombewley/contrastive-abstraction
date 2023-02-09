from __future__ import annotations
import numpy as np
from .utils import *


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

    def abstract_mapping(self, contexts: np.ndarray, states: np.ndarray, next_states: np.ndarray):
        num = contexts.shape[0]
        assert states.shape[0] == next_states.shape[0] == num, "Arrays must have common length."
        l_W_index = {w: i for i, w in enumerate(self.W.leaves)}
        l_X_index = {x: i for i, x in enumerate(self.X.leaves)}
        mapping = np.zeros((num, 3), dtype=np.int_)
        def _recurse(w, x, xx, indices):
            if len(indices) == 0: return
            w_l = w_r = w; x_l = x_r = x; xx_l = xx_r = xx
            try:
                right = w.sort(contexts[indices])
                w_l, w_r = w.left, w.right
            except AssertionError:
                try:
                    right = x.sort(states[indices])
                    x_l, x_r = x.left, x.right
                except AssertionError:
                    try:
                        right = xx.sort(next_states[indices])
                        xx_l, xx_r = xx.left, xx.right
                    except AssertionError:
                        mapping[indices] = [l_W_index[w], l_X_index[x], l_X_index[xx]]
                        return
            _recurse(w_l, x_l, xx_l, indices[~right])
            _recurse(w_r, x_r, xx_r, indices[ right])
        _recurse(self.W, self.X, self.X, np.arange(num))
        return mapping

    def transition_counts(self, contexts: np.ndarray, states: np.ndarray, next_states: np.ndarray):
        return mapping_to_counts(self.abstract_mapping(contexts, states, next_states), n=self.n, m=self.m)

    def eval_changes(self, contexts: np.ndarray, states: np.ndarray, next_states: np.ndarray,
                     context_split: bool = False, context_merge: bool = False,
                     state_split: bool = False, state_merge: bool = False,
                     exhaustive: bool = True, merge_pairs_only: bool = True, eps: float = 0., debug: bool = False):
        n, m = self.n, self.m
        context_split_gains, context_merge_gains, state_split_gains, state_merge_gains = None, None, None, None
        # Map the dataset through the extant abstractions and compute JSD as a baseline
        mapping_current = self.abstract_mapping(contexts, states, next_states)
        counts_current = mapping_to_counts(mapping_current, n=n, m=m)
        jsd_current = jsd(counts_current)[0]
        if context_split:
            assert m > 1, "Context splitting requires at least two abstract states."
            # Either exhaustively try all windows, or just the one with largest population
            windows_to_split = range(n) if exhaustive else [counts_current.sum(axis=(1, 2)).argmax()]
            context_split_gains = {w: {} for w in windows_to_split}
            for w in windows_to_split:
                parent = self.W.leaves[w]
                w_here_mask = mapping_current[:, 0] == w
                bounds = data_bounds(contexts[w_here_mask])
                # Add to window indices and expand counts arrays
                mapping_exp = mapping_current.copy()
                mapping_exp[mapping_exp[:, 0] > w, 0] += 1
                counts_exp = np.insert(counts_current, w + 1, 0, axis=0)
                for dim in range(self.d_c):
                    # Valid thresholds are within the bounds of the observations in this window + don't violate epsilon
                    th = self.context_split_thresholds[dim]
                    valid_thresholds = th[np.logical_and(
                        np.logical_and(bounds[0, dim] < th, parent.bounds[0, dim] + eps <= th),
                        np.logical_and(th < bounds[1, dim], parent.bounds[1, dim] - eps >= th))]
                    # Compare all context values to all valid thresholds in parallel
                    move = geq_threshold_mask(contexts, w_here_mask, dim, valid_thresholds)
                    move_any = move.any(axis=1)
                    # Build delta array
                    delta = build_delta_array_w(move[move_any], mapping_exp[move_any], w, n, m)
                    assert (delta.sum(axis=1) == 0).all()
                    # Record Jensen-Shannon divergence for each valid threshold
                    context_split_gains[w][dim] = (valid_thresholds, jsd(counts_exp + delta) - jsd_current)
                    # Check that computed counts match that after split is made
                    if debug and len(valid_thresholds) > 0:
                        greedy = np.argmax(context_split_gains[w][dim][1])
                        parent.split(dim, valid_thresholds[greedy])
                        match = (self.transition_counts(contexts, states, next_states)
                                 == (counts_exp + delta[greedy])).all()
                        assert match
                        parent.merge()
        if context_merge:
            assert n > 1, "Context merging requires at least two windows."
            context_merge_gains = []
            transition_sums = counts_current.sum(axis=0)
            for parent, ws in self.W.merge_sets(pairs_only=merge_pairs_only):
                # Merge counts along window axis
                counts_merged = np.concatenate([counts_current[:ws[0]],
                                                counts_current[ws].sum(axis=0, keepdims=True),
                                                counts_current[ws[-1] + 1:]], axis=0)
                assert (counts_merged.sum(axis=0) == transition_sums).all()
                # Record Jensen-Shannon divergence for each merge set
                context_merge_gains.append((parent, ws, jsd(counts_merged)[0] - jsd_current))
        if state_split:
            assert n > 1, "State splitting requires at least two context windows."
            # Either exhaustively try all states, or just the one with largest population (first states)
            states_to_split = range(m) if exhaustive else [counts_current.sum(axis=(0, 2)).argmax()]
            state_split_gains = {x: {} for x in states_to_split}
            for x in states_to_split:
                x_here_mask = mapping_current[:, 1] == x
                xx_here_mask = mapping_current[:, 2] == x
                bounds = data_bounds(states[x_here_mask], next_states[xx_here_mask])
                # Add to state indices and expand counts arrays
                mapping_exp = mapping_current.copy()
                mapping_exp[mapping_exp[:, 1] > x, 1] += 1
                mapping_exp[mapping_exp[:, 2] > x, 2] += 1
                counts_exp = np.insert(counts_current, x + 1, 0, axis=1)
                counts_exp = np.insert(counts_exp, x + 1, 0, axis=2)
                for dim in range(self.d_s):
                    # Valid thresholds are within the bounds of the observations in this abstract state
                    th = self.state_split_thresholds[dim]
                    valid_thresholds = th[np.logical_and(bounds[0, dim] < th, th < bounds[1, dim])]
                    # Compare all state values to all valid thresholds in parallel
                    move_x = geq_threshold_mask(states, x_here_mask, dim, valid_thresholds)
                    move_xx = geq_threshold_mask(next_states, xx_here_mask, dim, valid_thresholds)
                    move_any = np.logical_or(move_x, move_xx).any(axis=1)
                    # Build delta array
                    delta = build_delta_array_x(move_x[move_any], move_xx[move_any], mapping_exp[move_any], x, n, m)
                    assert (delta.sum(axis=(2, 3)) == 0).all()
                    # Record Jensen-Shannon divergence for each valid threshold
                    state_split_gains[x][dim] = (valid_thresholds, jsd(counts_exp + delta) - jsd_current)
                    # Check that computed counts match that after split is made
                    if debug and len(valid_thresholds) > 0:
                        greedy = np.argmax(state_split_gains[x][dim][1])
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
            for parent, xs in self.X.merge_sets(pairs_only=merge_pairs_only):
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
                           context_split: bool = False, context_merge: bool = False, state_split: bool = False,
                           state_merge: bool = False, exhaustive: bool = True, merge_pairs_only: bool = False,
                           alpha: float = 0., beta: float = 0., power: float = 1., eps: float = 0., tau: float = 0.):
        n, m = self.n, self.m
        context_split_gains, context_merge_gains, state_split_gains, state_merge_gains = self.eval_changes(
            contexts, states, next_states,
            context_split=context_split and m > 1, state_split=state_split and n > 1,
            context_merge=context_merge and m > 1 and n > 1, state_merge=state_merge and m > 1 and n > 1,
            exhaustive=exhaustive, merge_pairs_only=merge_pairs_only, eps=eps)
        candidates, quals = [], []
        if context_split and m > 1:
            for w in context_split_gains:
                for dim in range(self.d_c):
                    g = context_split_gains[w][dim]
                    if len(g[1]) > 0:
                        greedy = np.argmax(g[1])
                        qual = g[1][greedy] - beta * (n**power - (n - 1)**power)
                        if qual > 0:
                            candidates.append(("context split", w, dim, g[0][greedy]))
                            quals.append(qual)
        if state_split and n > 1:
            for x in state_split_gains:
                for dim in range(self.d_s):
                    g = state_split_gains[x][dim]
                    if len(g[1]) > 0:
                        greedy = np.argmax(g[1])
                        qual = g[1][greedy] - alpha * (m**power - (m - 1)**power)
                        if qual > 0:
                            candidates.append(("state split", x, dim, g[0][greedy]))
                            quals.append(qual)
        if context_merge and m > 1 and n > 1:
            for parent, ws, gain in context_merge_gains:
                qual = gain - beta * ((n - len(ws))**power - (n - 1)**power)
                if qual > 0:
                    candidates.append(("context merge", parent, ws))
                    quals.append(qual)
        if state_merge and m > 1 and n > 1:
            for parent, xs, gain in state_merge_gains:
                qual = gain - alpha * ((m - len(xs))**power - (m - 1)**power)
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
                print(f"(n={self.n}, m={self.m}) Split window {w} at {self.W.dims[dim]}={threshold}")
            elif candidates[chosen][0] == "context merge":
                _, parent, ws = candidates[chosen]
                parent.merge()
                print(f"(n={self.n}, m={self.m}) Merge windows {ws}")
            elif candidates[chosen][0] == "state split":
                _, x, dim, threshold = candidates[chosen]
                self.X.leaves[x].split(dim, threshold)
                print(f"(n={self.n}, m={self.m}) Split abstract state {x} at {self.X.dims[dim]}={threshold}")
            elif candidates[chosen][0] == "state merge":
                _, parent, xs = candidates[chosen]
                parent.merge()
                print(f"(n={self.n}, m={self.m}) Merge abstract states {xs}")
        return context_split_gains, context_merge_gains, state_split_gains, state_merge_gains, candidates, quals


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
        return " and ".join([("" if np.isinf(l) else f"{l:.2f}<=") + dim + ("" if np.isinf(u) else f"<{u:.2f}")
                             for dim, (l, u) in zip(self.dims, self.bounds.T) if not (np.isinf(l) and np.isinf(u))])

    @property
    def is_leaf(self):
        return self.split_dim is None

    @property
    def leaves(self):
        if self.is_leaf:
            return [self]
        return self.left.leaves + self.right.leaves

    def merge_sets(self, pairs_only: bool = False):
        leaves = self.leaves
        merge_sets_ = []
        def _recurse(subset):
            if subset.is_leaf: return
            if not(pairs_only) or (subset.left.is_leaf and subset.right.is_leaf):
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
        assert self.bounds[0, dim] <= threshold < self.bounds[1, dim], f"Split (dim {dim}={threshold}) out of bounds."
        self.split_dim = dim
        self.split_threshold = threshold
        bounds_left, bounds_right = self.bounds.copy(), self.bounds.copy()
        bounds_left[1, dim] = bounds_right[0, dim] = threshold
        self.left = HRSubset(self.dims, bounds_left)
        self.right = HRSubset(self.dims, bounds_right)

    def multisplit(self, dim: int, thresholds):
        self.split(dim, thresholds[0])
        if len(thresholds) > 1:
            self.right.multisplit(dim, thresholds[1:])

    def show(self, ax, dims, bounds, nums=False, lw=0.5, alpha=1, zorder=-1):
        for i, leaf in enumerate(self.leaves):
            leaf_bounds = leaf.bounds[:, dims]
            leaf_bounds[0] = np.maximum(leaf_bounds[0], bounds[0])
            leaf_bounds[1] = np.minimum(leaf_bounds[1], bounds[1])
            rect = bounds_to_rect(leaf_bounds, fill_colour=None, lw=lw, alpha=alpha, zorder=zorder)
            ax.add_patch(rect)
            if nums:
                x, y = rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2
                ax.text(x, y, i, horizontalalignment="center", verticalalignment="center")
        ax.autoscale_view()
        return ax
