import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from csta.csta import ContrastiveAbstraction, jsd


df = pd.read_csv(f"data/maze-RL.csv")
t = df["time"].values
not_first = np.argwhere(t > 0).flatten()

contexts = df[["ep", "time"]].values[not_first]
all_states = df[["x", "y"]].values
states = all_states[not_first - 1]
next_states = all_states[not_first]

model = ContrastiveAbstraction(
    context_dims=["ep", "time"],
    context_split_thresholds=[np.arange(1, 750), np.array([])],
    state_dims=["x", "y"],
    state_split_thresholds=[np.linspace(0.1, 9.9, 100), np.linspace(0.1, 9.9, 100)],
    )

fig = plt.figure()
ax = plt.axes(projection="3d")

model.X.merge()
model.set_1d_context_windows(375)
history = []
for _ in range(10):
    model.make_greedy_change(contexts, states, next_states, alpha=0., beta=0., tau=0.)
    history.append((model.n, model.m,
                    jsd(model.transition_mask(contexts, states, next_states).sum(axis=3))[0]))

ax.plot3D(*np.array(history).T, c="g")
ax.set_xlabel("Number of context windows")
ax.set_ylabel("Number of abstract states")
ax.set_zlabel("Jensen-Shannon divergence")

# num_axes = len(gains)
# num_rows = int(np.sqrt(num_axes))
# num_cols = int(np.ceil(num_axes / num_rows))
# _, axes = plt.subplots(num_rows, num_cols, sharey=True, squeeze=False)
# axes = axes.flatten()
# for x, gain_x in enumerate(gains):
#     for d, (thresholds_x_d, gain_x_d) in enumerate(gain_x):
#         axes[x].plot(thresholds_x_d, gain_x_d)

plt.show()
