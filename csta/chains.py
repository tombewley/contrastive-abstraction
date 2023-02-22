from __future__ import annotations
import numpy as np
import networkx as nx
from .utils import powerset


class MarkovChains:
    """
    Class for manipulating a set of n Markov chains over a common set of m states.
    which are defined by an n x m x m array of counts or conditional transition probabilities.
    """
    def __init__(self, states: list, array: np.ndarray):
        assert len(array.shape) == 3 and array.shape[1] == array.shape[2] == len(states), "Misshapen counts/probs array."
        self.states = states
        if np.isclose(array.sum(axis=2), 1).all():  # Interpret array as conditional
            self.counts = None
            self.conditional = array
        else:
            self.counts = array
            self.conditional = self.counts / self.counts.sum(axis=2, keepdims=True)

    @property
    def joint(self):
        return self.counts / self.counts.sum(axis=(1, 2), keepdims=True)

    @property
    def marginal(self):
        sums = self.counts.sum(axis=2)
        return sums / sums.sum(axis=1, keepdims=True)

    @property
    def stationary(self):
        # Adapted from https://stackoverflow.com/a/58334399
        P = self.conditional.copy()
        P[np.isnan(P)] = 0.
        evals, evecs = np.linalg.eig(np.swapaxes(P, 1, 2))
        eval1 = np.isclose(evals, 1)
        num_eval1 = eval1.sum(axis=1)
        no_unique_eval1 = num_eval1 != 1
        if no_unique_eval1.any():
            raise ValueError(f"Chains {np.argwhere(no_unique_eval1).flatten()} have no unique stationary distribution")
        evec1 = np.swapaxes(evecs, 1, 2)[eval1]
        return (evec1 / evec1.sum(axis=1, keepdims=True)).real

    @property
    def fundamental(self):
        n, m, _ = self.conditional.shape
        N_and_B = np.tile(np.expand_dims(np.identity(m), 0), (n, 1, 1))
        for i, P in enumerate(self.conditional):
            transient = ~np.isnan(P).any(axis=1)
            transient_idx = np.argwhere(transient).flatten()
            n_tr = transient.sum()
            N = np.linalg.inv(np.identity(n_tr) - P[transient][:, transient])  # Visits before absorption
            B = np.matmul(N, P[transient][:, ~transient])  # Absorption probabilities
            for x, N_row, B_row in zip(transient_idx, N, B):
                N_and_B[i, x, transient] = N_row
                N_and_B[i, x, ~transient] = B_row
        return N_and_B

    def perturb(self, source: list[tuple], dest: list[tuple]):
        perturbed_conditional = self.conditional.copy()
        for s, d in zip(source, dest):
            perturbed_conditional[d] = self.conditional[s]
        return MarkovChains(states=self.states, array=perturbed_conditional)

    def powerset_perturb(self, source: int, dest: int):
        m = len(self.states)
        # NOTE: when one (and only one) of source and dest have NaNs on a row, always use the other one
        source_nanfix, dest_nanfix = self.conditional[source].copy(), self.conditional[dest].copy()
        source_isnan = np.isnan(source_nanfix).any(axis=1)
        dest_isnan = np.isnan(dest_nanfix).any(axis=1)
        source_nanfix[source_isnan] = dest_nanfix[source_isnan]
        dest_nanfix[dest_isnan] = source_nanfix[dest_isnan]
        perturbed_conditional = np.repeat(source_nanfix[None, :], 2**m, axis=0)
        dest_nanfix = dest_nanfix[None, :]  # Needed for indexing with xs
        for i, xs in enumerate(powerset(range(m))):
            perturbed_conditional[i, xs] = dest_nanfix[0, xs]
        return MarkovChains(states=self.states, array=perturbed_conditional)

    def show_graph(self, c):
        # TODO: Pull out into separate class and bring over method from previous implementation
        graph = nx.DiGraph()
        for i, x in enumerate(self.states):
            for j, xx in enumerate(self.states):
                graph.add_edge(x, xx, counts=self.counts[:, i, j])
        pos = nx.spring_layout(graph)
        width = [5 * ((d["counts"][c] / self.counts.max())**0.5) for _, _, d in graph.edges(data=True)]
        nx.draw(graph, pos=pos, with_labels=True,
                width=width,
                connectionstyle="arc3,rad=0.2"
                )
