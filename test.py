import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, psutil

from csta.csta import ContrastiveAbstraction, jsd


np.set_printoptions(suppress=True, precision=3)
process = psutil.Process(os.getpid())

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
ax.set_xlabel("Number of context windows")
ax.set_ylabel("Number of abstract states")
ax.set_zlabel("Jensen-Shannon divergence")

alpha = 0.01
beta = 0.02

for batch_size in (len(states), ):

    model.X.merge()
    model.set_1d_context_windows(375)
    history = []
    for i in range(200):
        print(process.memory_info().rss / 1048576, "MB")
        batch = np.random.choice(len(states), size=batch_size, replace=False)

        candidates = model.make_greedy_change(contexts[batch], states[batch], next_states[batch],
                                              alpha=alpha, beta=beta, tau=0.)
        if len(candidates) == 0: break
        history.append((model.n, model.m,
                        jsd(model.transition_mask(contexts, states, next_states).sum(axis=3))[0]))
        print(i, history[-1])

    ax.plot3D(*np.array(history).T, label=batch_size)

for x in model.X.leaves:
    print(x)

N = model.transition_mask(contexts, states, next_states).sum(axis=3)
P = N / N.sum(axis=(1, 2), keepdims=True)
print(N)
print(P)

plt.legend()

# num_axes = len(gains)
# num_rows = int(np.sqrt(num_axes))
# num_cols = int(np.ceil(num_axes / num_rows))
# _, axes = plt.subplots(num_rows, num_cols, sharey=True, squeeze=False)
# axes = axes.flatten()
# for x, gain_x in enumerate(gains):
#     for d, (thresholds_x_d, gain_x_d) in enumerate(gain_x):
#         axes[x].plot(thresholds_x_d, gain_x_d)

plt.show()
