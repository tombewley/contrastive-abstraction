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
        evals, evecs = np.linalg.eig(np.swapaxes(self.conditional, 1, 2))
        evec1 = np.swapaxes(evecs, 1, 2)[np.isclose(evals, 1)]
        return (evec1 / evec1.sum(axis=1, keepdims=True)).real

    def perturb(self, source: list[tuple], dest: list[tuple]):
        perturbed_conditional = self.conditional.copy()
        for s, d in zip(source, dest):
            perturbed_conditional[d] = self.conditional[s]
        return MarkovChains(states=self.states, array=perturbed_conditional)

    def powerset_perturb(self, source: int, dest: int):
        m = len(self.states)
        perturbed_conditional = np.repeat(self.conditional[source:source+1], 2**m, axis=0)
        for i, xs in enumerate(powerset(range(m))):
            perturbed_conditional[i, xs] = self.conditional[dest, xs]
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
