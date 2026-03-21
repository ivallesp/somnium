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
df = pd.read_csv(os.path.join(DATA_DIR, "creditcard.csv"))

# Separate labels and features (drop Time, keep V1-V28 + Amount)
labels = df["Class"].values
feature_names = [c for c in df.columns if c not in ("Time", "Class")]
data = df[feature_names].values

# Subsample for tractability (keep all fraud, sample normal)
fraud_idx = np.where(labels == 1)[0]
normal_idx = np.where(labels == 0)[0]
rng = np.random.RandomState(42)
normal_sample = rng.choice(normal_idx, size=5000, replace=False)
idx = np.concatenate([fraud_idx, normal_sample])
rng.shuffle(idx)

data = data[idx]
labels = labels[idx]
names = feature_names

print(f"Data shape: {data.shape}  (fraud: {labels.sum():.0f}, normal: {(labels == 0).sum()})")

# %% Train SOM
model = SOM(lattice="hexa", normalization="standard", distance_metric="euclidean",
            neighborhood="gaussian", mapsize=[20, 20], n_jobs=1)

model.fit(data, 30, 20, 5)
model.fit(data, 30, 5, 1)

print("E_Quantization =", model.calculate_quantization_error())
print("E_Topographic =", model.calculate_topographic_error())

if PLOT:
    plot_components(model, names, figure_width=30, max_subplot_columns=5)
    plot_bmus(model, figure_width=10)
    plot_umatrix(model, colormap=plt.cm.hot, figure_width=10)
    plt.show()
