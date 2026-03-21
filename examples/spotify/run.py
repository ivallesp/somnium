import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from somnium.core import SOM
from somnium.visualization import plot_components, plot_bmus, plot_umatrix

PLOT = True

# %% Prepare data
subprocess.run(["python", os.path.join(os.path.dirname(__file__), "prepare.py")], check=True)

# %% Load data
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))

# Continuous audio features
feature_names = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "popularity",
]

df = df.dropna(subset=feature_names)

# Subsample for tractability
rng = np.random.RandomState(42)
idx = rng.choice(len(df), size=10000, replace=False)
df_sample = df.iloc[idx]

data = df_sample[feature_names].values
genres = df_sample["track_genre"].values
names = feature_names

print(f"Data shape: {data.shape}")
print(f"Genres: {len(set(genres))} unique")

# %% Train SOM
model = SOM(lattice="hexa", normalization="standard", distance_metric="euclidean",
            neighborhood="gaussian", mapsize=[20, 20], n_jobs=1)

model.fit(data, 30, 20, 5)
model.fit(data, 30, 5, 1)

print("E_Quantization =", model.calculate_quantization_error())
print("E_Topographic =", model.calculate_topographic_error())

if PLOT:
    plot_components(model, names, figure_width=30, max_subplot_columns=4)
    plot_bmus(model, figure_width=10)
    plot_umatrix(model, colormap=plt.cm.hot, figure_width=10)
    plt.show()
