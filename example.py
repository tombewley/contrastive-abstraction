"""
Example
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csta


FNAME = "maze-RL"
DIM_NAMES = ["x","y"]#,"vx","vy"]

M_MAX = np.inf
ALPHA = 0.05

TIME_ABSTRACT = True
WINDOWS = None # [[0,100]] # NOTE: Manual windows take priority over n_max, beta, epsilon.
N_MAX = np.inf
BETA = 0.01
EPSILON = 15 # Minimum window width

PLOT = True
PLOT_DIMS = [0, 1]
MAX_EPS_TO_PLOT = 50

SAVE = False


# Load data.
df = pd.read_csv(f"data/{FNAME}.csv")
D = df[DIM_NAMES].values

print(df)

# Use uniformly-spaced range of split thresholds.
thresholds = [np.arange(0.1, 10, 0.1), np.arange(0.1, 10, 0.1)]
# Use quantiles as split thresholds, with double use of unique() function and midpoint interpolation to handle categoricals.
# thresholds = csta.quantile_thresholds(D, q=1)

# Use episode numbers as class_data.
class_data = df["ep"].values 
term_data = np.array(list(df["time"] == 0)[1:] + [True]) 

tree = csta.Tree(dim_names=DIM_NAMES, D=D, term_data=term_data, class_data=class_data, thresholds=thresholds)
tree.state_abstract(M_MAX, ALPHA, verbose_level=1, plot_dims=(PLOT_DIMS if PLOT else None), max_eps_to_plot=MAX_EPS_TO_PLOT)
tree.show([0,1], lw=0.5)

if TIME_ABSTRACT: tree.time_abstract(windows=WINDOWS, n_max=N_MAX, beta=BETA, epsilon=EPSILON, verbose_level=1, plot=PLOT, validate=True)

for p in [0,1]:
    g = csta.Graph(tree.C[:,:,p], [l.bb for l in tree.leaves])
    # g.show(layout_dims=[0,1], terminal=False, ax=ax)
    g.show(terminal=True)

plt.show()