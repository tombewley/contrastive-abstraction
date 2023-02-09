"""
(Approximately) replicates the abstraction in Figure 1 of
"Summarising and Comparing Agent Dynamics with Contrastive Spatiotemporal Abstraction".
NOTE: The final state split differs slightly due to small mathematical changes inside the model.
"""
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from csta.model import ContrastiveAbstraction
from csta.utils import jsd


# Parse optional arguments
parser = ArgumentParser()
parser.add_argument("-a", "--alpha", type=float, default=0.05)  # Regularisation on umber of abstract states (m)
parser.add_argument("-b", "--beta", type=float, default=0.01)  # Regularisation on number of time windows (n)
parser.add_argument("-e", "--epsilon", type=float, default=25)  # Minimum time window width
parser.add_argument("-ph", "--plot_history", type=int, default=0)
parser.add_argument("-pg", "--plot_gains", type=int, default=0)
args = parser.parse_args()

# Read dataset
# Context (i.e. values used for temporal abstraction) = episode number
# State dimensions  = x position, y position
df = pd.read_csv(f"data/maze.csv")
t = df["time"].values
not_first = np.argwhere(t > 0).flatten()
contexts = df[["ep"]].values[not_first]
all_states = df[["x", "y"]].values
states = all_states[not_first - 1]
next_states = all_states[not_first]

# Allow spatial splits to be made at increments of 0.1 along both dimensions
state_split_thresholds = [np.arange(0.1, 10, 0.1), np.arange(0.1, 10, 0.1)]

# Allow temporal splits to be made at all unique episode numbers (excluding zero)
episode_split_thresholds = np.unique(contexts[:, 0])[1:]

# Initialise the model
model = ContrastiveAbstraction(context_dims=["ep"],
                               context_split_thresholds=[episode_split_thresholds],
                               state_dims=["x", "y"],
                               state_split_thresholds=state_split_thresholds
                               )

# Prior to state abstraction stage, initialise with 'maximal' temporal abstraction using all split points
model.set_1d_context_windows(*episode_split_thresholds)

# Start by considering state splits only
do_state_splits, do_episode_splits = True, False

# Set up optional 3D plot of JSD against m and n
if args.plot_history:
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_ylabel("Number of abstract states (m)")
    ax.set_xlabel("Number of time windows (n)")
    ax.set_zlabel("Jensen-Shannon divergence")

i, history = 0, []
while True:  # Currently have no limit on total number of iterations
    print(f"Iteration {i}:")

    # Identify the greedy split on either a state dimension (first stage) or the episode dimension (second stage)
    context_split_gains, context_merge_gains, state_split_gains, state_merge_gains, candidates, quals = \
        model.make_greedy_change(contexts, states, next_states,
                                 context_split=do_episode_splits, state_split=do_state_splits,
                                 alpha=args.alpha, beta=args.beta, eps=args.epsilon)

    # Optionally plot JSD gains of all splits attempted
    if args.plot_gains:
        for gains, W_or_X, w_or_x_str in zip(
                [context_split_gains, state_split_gains], [model.W, model.X], ["time window", "abstract state"]):
            if gains is None:
                continue
            num_axes = len(gains)
            num_rows = int(np.sqrt(num_axes))
            num_cols = int(np.ceil(num_axes / num_rows))
            _, axes = plt.subplots(num_rows, num_cols, sharey=True, squeeze=False)
            axes = axes.flatten()
            for axx, (w_or_x, gain_w_or_x) in zip(axes, gains.items()):
                for dim, (thresholds_w_or_x_d, gain_w_or_x_d) in gain_w_or_x.items():
                    axx.plot(thresholds_w_or_x_d, gain_w_or_x_d, label=W_or_X.dims[dim])
                    axx.set_title(f"Splitting {w_or_x_str} {w_or_x}")
            plt.suptitle(f"Iteration {i}")
            plt.legend()

    if len(candidates) == 0:  # This occurs when no sufficiently good splits are found
        if do_state_splits:
            # Move to second stage of splitting along episode dimension
            do_state_splits, do_episode_splits = False, True
            # Start this stage by merging all windows into one
            model.W.merge()
        else:
            # Terminate second stage
            break

    # Compute latest JSD
    jsd_ = jsd(model.transition_counts(contexts, states, next_states))[0]
    print(f"JSD={jsd_}")
    if args.plot_history:
        history.append((model.m, model.n, jsd_))

    i += 1

# Populate 3D plot of JSD against m and n
if args.plot_history:
    ax.plot3D(*np.array(history).T, "-o")

# Plot final abstractions
_, ax = plt.subplots()
model.X.show(ax=ax, dims=[0, 1], bounds=np.array([[0, 0], [10, 10]]), nums=True)
ax.set_title("State Abstraction")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
_, ax = plt.subplots()
model.W.show(ax=ax, dims=[0], bounds=np.array([[0], [750]]), nums=True)
ax.set_title("Time Windows")
ax.set_xlabel("Episode number")

plt.show()
