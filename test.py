import pandas as pd
import matplotlib.pyplot as plt

from csta.csta import *


df = pd.read_csv(f"data/maze-RL.csv")
t = df["time"].values
not_first = np.argwhere(t > 0).flatten()

timestamps = df[["ep", "time"]].values[not_first]
all_states = df[["x", "y"]].values
states = all_states[not_first - 1]
next_states = all_states[not_first]

model = CSTA(spatial_dims=["x", "y"], temporal_dims=["ep", "time"])

model.T.split(0, 375)

qual = model.eval_splits(timestamps, states, next_states)


plt.plot(qual[0][0])
plt.plot(qual[0][1])
plt.show()
