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

# %% Load and harmonize all years
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

COLUMN_MAP = {
    "country": {
        "Country", "Country or region", "Country name",
    },
    "score": {
        "Happiness Score", "Happiness.Score", "Score", "Ladder score", "Happiness score",
    },
    "gdp": {
        "Economy (GDP per Capita)", "Economy..GDP.per.Capita.", "GDP per capita",
        "Logged GDP per capita", "Explained by: GDP per capita",
    },
    "social_support": {
        "Family", "Social support", "Explained by: Social support",
    },
    "health": {
        "Health (Life Expectancy)", "Health..Life.Expectancy.",
        "Healthy life expectancy", "Explained by: Healthy life expectancy",
    },
    "freedom": {
        "Freedom", "Freedom to make life choices",
        "Explained by: Freedom to make life choices",
    },
    "generosity": {
        "Generosity", "Explained by: Generosity",
    },
    "corruption": {
        "Trust (Government Corruption)", "Trust..Government.Corruption.",
        "Perceptions of corruption", "Explained by: Perceptions of corruption",
    },
}

frames = []
for year in range(2015, 2023):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{year}.csv"))
    row = {}
    for target, sources in COLUMN_MAP.items():
        for col in df.columns:
            if col in sources:
                row[target] = df[col]
                break
    out = pd.DataFrame(row)
    out["year"] = year
    frames.append(out)

df = pd.concat(frames, ignore_index=True)

feature_names = ["gdp", "social_support", "health", "freedom", "generosity", "corruption"]
df = df.dropna(subset=feature_names)
for col in feature_names:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
df = df.dropna(subset=feature_names)
data = df[feature_names].values.astype(float)
countries = df["country"].values

print(f"Data shape: {data.shape}  ({df['year'].nunique()} years, {df['country'].nunique()} countries)")

# %% Train SOM
model = SOM(lattice="hexa", normalization="standard", distance_metric="euclidean",
            neighborhood="gaussian", mapsize=[15, 10], n_jobs=1)

model.fit(data, 30, 20, 5)
model.fit(data, 30, 5, 1)

print("E_Quantization =", model.calculate_quantization_error())
print("E_Topographic =", model.calculate_topographic_error())

if PLOT:
    plot_components(model, feature_names, figure_width=30, max_subplot_columns=3)
    plot_bmus(model, figure_width=10)
    plot_umatrix(model, colormap=plt.cm.hot, figure_width=10)
    plt.show()
