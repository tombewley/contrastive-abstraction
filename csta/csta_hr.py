import hyperrectangles as hr
import numpy as np
import bisect


class Model:
    def __init__(self, dim_names, data:np.ndarray, labels:np.ndarray, nonterminal:np.ndarray):
        self.tree = hr.tree.Tree(name="state_abstraction", 
                                 root=hr.node.Node(hr.Space(
                                     dim_names=dim_names+["label"], 
                                     data=np.hstack((data, labels.reshape(-1,1)))
                                    )),
                                 split_dims=[], eval_dims=[] # Internal split methods not used
                                 ).populate()
        self.labels = labels
        self.window_starts = np.array([])
        assert nonterminal[-1] == 0
        self.nonterminal = nonterminal
        self.noninitial = np.zeros_like(self.nonterminal)
        self.noninitial[1:] = self.nonterminal[:-1]

    @property
    def m(self): return len(self.tree)
    @property
    def n(self): return len(self.window_starts)+1

    @property
    def w_x_xn(self):
        """Return array containing [w, current x, next x] for each sample"""
        w_x_xn = np.full((len(self.tree.space.data),3), self.m, dtype=int)
        w_x_xn[:,0] = np.searchsorted(self.window_starts, self.labels, side="right")
        for x in range(self.m): w_x_xn[self.tree.leaves[x].sorted_indices[:,0],1] = x
        w_x_xn[self.nonterminal,2] = w_x_xn[self.noninitial,1]
        return w_x_xn

    @property
    def counts(self):
        """Return n x (m+1) x (m+1) counts array"""
        counts = np.zeros((self.n, self.m+1, self.m+1), dtype=int)
        i, c = np.unique(self.w_x_xn, axis=0, return_counts=True)
        i1, i2, i3 = i.T # NOTE: Python 3.11 allows unpacking inside subscript
        counts[i1, i2, i3] = c
        return counts

    def counts_delta(self, x, d, s, s_prev=None):
        """Compute the counts delta induced by splitting x along d at s,
        or by *moving* the split boundary rightwards from s_prev to s"""
        samples = self.tree.leaves[x].sorted_indices[:,d]
        data = self.tree.space.data[samples,d]
        idx_min = 0 if s_prev is None else bisect.bisect(data, s_prev)
        idx_max = bisect.bisect(data, s)
        print(idx_min, idx_max)
        left, moved, right = np.split(samples, [idx_min, idx_max])
        print(moved)
        pred = moved[self.noninitial[moved]]-1
        succ = moved[self.nonterminal[moved]]+1
        # Seven kinds of update to perform:
        # Inward:
        #   (left, right) to (left, left)
        #   (right, right) to (right, left)
        #   (right, right) to (left, left)
        #   (other, right) to (other, left)
        # Outward:
        #   (right, left) to (left, left)
        #   (right, right) to (left, right)
        #   (right, right) to (left, left) *REPEAT
        #   (right, other) to (left, other)
