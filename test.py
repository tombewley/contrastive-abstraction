import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from csta.csta import CSTA


df = pd.read_csv(f"data/maze-RL.csv")
t = df["time"].values
not_first = np.argwhere(t > 0).flatten()

timestamps = df[["ep", "time"]].values[not_first]
all_states = df[["x", "y"]].values
states = all_states[not_first - 1]
next_states = all_states[not_first]

model = CSTA(temporal_dims=["ep", "time"],
             temporal_split_thresholds=[np.arange(1, 750), np.array([])],
             spatial_dims=["x", "y"],
             spatial_split_thresholds=[np.linspace(0.1, 9.9, 100), np.linspace(0.1, 9.9, 100)],
             )

model.set_windows(*np.arange(1, max(df["ep"])))
# model.set_windows(375)

model.make_greedy_change(timestamps, states, next_states)
model.make_greedy_change(timestamps, states, next_states)
model.make_greedy_change(timestamps, states, next_states)
gain = model.make_greedy_change(timestamps, states, next_states)

for x in model.X.leaves:
    print(x)

# m = model.m - 1
# num_rows = int(np.sqrt(m))
# num_cols = int(np.ceil(m / num_rows))
# _, axes = plt.subplots(num_rows, num_cols, sharey=True, squeeze=False)
# axes = axes.flatten()
# for x, gain_x in enumerate(gain):
#     for d, (thresholds_x_d, gain_x_d) in enumerate(gain_x):
#         axes[x].plot(thresholds_x_d, gain_x_d)
#
# plt.show()
