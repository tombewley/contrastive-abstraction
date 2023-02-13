import numpy as np
from numba import jit
from itertools import chain, combinations
from math import factorial
from matplotlib.patches import Rectangle


def quantile_thresholds(data, q=1, method="higher"):
    # Double use of unique() function and higher interpolation handles categoricals
    # NOTE: using method="midpoint" seems to create precision issues during splitting process
    return [np.unique(np.percentile(np.unique(data[:, d]), q=np.arange(q, 100, q), method=method))
            for d in range(data.shape[1])]


def mapping_to_counts(mapping, n, m):
    counts = np.zeros((n, m, m), dtype=np.int_)
    for (w, x, xx), c in zip(*np.unique(mapping, axis=0, return_counts=True)):
        counts[w, x, xx] = c
    return counts


def data_bounds(*data):
    data = np.concatenate(data)
    if data.shape[0] == 0:
        return np.array([np.full(data.shape[1], -np.inf), np.full(data.shape[1], np.inf)])
    return np.array([data.min(axis=0), data.max(axis=0)])


@jit(nopython=True, cache=True)
def build_delta_array_w(contexts, w_here_mask, mapping_exp, dim, thresholds, w, n, m):
    # Get index of first threshold strictly > each context along the specified dimension
    first_threshold = np.zeros(len(contexts), dtype=np.int_)
    first_threshold[w_here_mask] = np.searchsorted(thresholds, contexts[w_here_mask, dim], side="right")
    # Iterate through data to update delta
    delta = np.zeros((len(thresholds), n + 1, m, m), dtype=np.int_)
    for ft, (w, xi, xxi) in zip(first_threshold, mapping_exp):
        if ft == 0: continue
        delta[:ft, w, xi, xxi] -= 1
        delta[:ft, w + 1, xi, xxi] += 1
    return delta


@jit(nopython=True, cache=True)
def build_delta_array_x(states, next_states, x_here_mask, xx_here_mask, mapping_exp, dim, thresholds, x, n, m):
    # Get index of first threshold strictly > each state and next state along the specified dimension
    first_threshold = np.zeros((len(states), 2), dtype=np.int_)
    first_threshold[x_here_mask, 0] = np.searchsorted(thresholds, states[x_here_mask, dim], side="right")
    first_threshold[xx_here_mask, 1] = np.searchsorted(thresholds, next_states[xx_here_mask, dim], side="right")
    # Iterate through data to update delta
    delta = np.zeros((len(thresholds), n, m + 1, m + 1), dtype=np.int_)
    for ft, mn_i, (w, xi, xxi) in zip(first_threshold, np.argmin(first_threshold, axis=1), mapping_exp):
        mn_v, mx_v = ft[mn_i], ft[1 - mn_i]
        if mx_v == 0: continue
        if mn_i == 1:
            # Handle transitions where the first state is in the right child
            delta[mn_v:mx_v, w, x, xxi] -= 1
            delta[mn_v:mx_v, w, x + 1, xxi] += 1
        # Handle transitions where the second state is in the right child
        else:
            delta[mn_v:mx_v, w, xi, x] -= 1
            delta[mn_v:mx_v, w, xi, x + 1] += 1
        if mn_v == 0: continue
        # Handle transitions where both states are in the right child
        delta[:mn_v, w, x, x] -= 1
        delta[:mn_v, w, x + 1, x + 1] += 1
    return delta


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


def bounds_to_rect(bounds, fill_colour=None, edge_colour="k", alpha=1, lw=0.5, zorder=-1):
    xl, xu = bounds[:, 0]
    yl, yu = bounds[:, 1] if bounds.shape[1] == 2 else (0, 1)
    fill_bool = (fill_colour != None)
    return Rectangle(xy=[xl, yl], width=xu-xl, height=yu-yl,
                     fill=fill_bool, facecolor=fill_colour, alpha=alpha, edgecolor=edge_colour, lw=lw, zorder=zorder)


@jit(nopython=True, cache=True)
def softmax(x, tau):
    x = x / tau
    x = x - x.max()  # For stability
    return np.exp(x) / np.exp(x).sum()


def print_matrix(model, matrix):
    float_prec = np.get_printoptions()["precision"]
    def float_formatter(f):
        s = f"{{:.{float_prec}f}}".format(f)
        z = len(s) - len(s.rstrip("0"))
        return s.rstrip("0") + " " * z
    with np.printoptions(formatter={"int": f"{{:>{len(str(int(np.nanmax(matrix))))}d}}".format,
                                    "float": float_formatter}):
        for w, mat_w in zip(model.W.leaves, matrix.copy()):
            print(w)
            for x, mat_w_x in zip(model.X.leaves, mat_w):
                print("    ", mat_w_x, x)


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def shapley(v: dict):
    contrib = {}
    for xs in v:
        for i, x in enumerate(xs):
            other_xs = xs - {x}
            if x not in contrib:
                contrib[x] = dict()
            contrib[x][other_xs] = v[xs] - v[other_xs]
    n = len(contrib)
    # Compute Shapley values
    n_fact = factorial(n)
    w = [factorial(i) * factorial(n - i - 1) / n_fact for i in range(0, n)]
    shap = {x: sum(w[len(other_xs)] * cont[other_xs]  # weighted sum of contributions...
                   for other_xs in cont)              # starting from each coalition of other features...
            for x, cont in contrib.items()}           # for each feature x
    # Compute pairwise interaction effects
    two_nm1_fact = 2 * factorial(n - 1)
    w_int = [factorial(i) * factorial(n - i - 2) / two_nm1_fact for i in range(0, n - 1)]
    shap_int = {xi: {xj: sum(w_int[len(other_xs)] * (cont_i[other_xs | {xj}] - cont_i[other_xs])  # weighted sum of increases in xi's contribution when xj is added
                             for other_xs in cont_i if xj not in other_xs)                        # starting from each coalition of other features...
                     for xj in contrib if xj != xi} for xi, cont_i in contrib.items()}            # for each pair of distinct features xi, xj
    # Compute remaining effects
    for x in contrib:
        shap_int[x][x] = shap[x] - sum(shap_int[x].values())
    return shap, shap_int
