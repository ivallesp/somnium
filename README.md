# _Somnium_: flexible Self-Organising Maps implementation

## What is it?
Somnium is a library for exploring multi-dimensional datasets using the Self-Organising Map algorithm (aka Kohonen map).

A _Self-Organising Map_ (_SOM_) is a biologically inspired algorithm for exploring multi-dimensional non-linear relations between variables. SOM was proposed in 1984 by Teuvo Kohonen. It compresses high-dimensional data into geometric relationships on a low-dimensional grid.

## Main applications
- Discover non-linear relations between variables at a glance.
- Micro-segment instances in a visually understandable way.
- Work as a surrogate model of a black-box model for explainability.
- Feature selection/reduction by finding correlated variables.

## Installation

```bash
git clone https://github.com/ivallesp/somnium
cd somnium
uv sync
```

## Quick start

```python
from somnium.core import SOM

model = SOM(
    mapsize="auto",           # or (15, 10)
    lattice="hexa",           # "hexa" or "rect"
    neighborhood="gaussian",  # "gaussian", "bubble", "cut_gaussian", "epanechicov"
    normalization="standard", # "standard", "minmax", "log", "logistic", "boxcox"
    initialization="pca",     # "pca" or "random"
    distance_metric="euclidean",
)

# Two-phase training (rough + fine)
model.fit(data, epochs=30, radiusin=20, radiusfin=5)
model.fit(data, epochs=30, radiusin=5,  radiusfin=1)

# Or use the convenience method
model.fit_auto(data)

# Metrics (lower is better for all)
model.calculate_quantization_error()
model.calculate_topographic_error()
model.calculate_vacancy_rate()

# Predict BMUs for new data
bmus = model.predict(new_data)
```

## Visualization

```python
from somnium.visualization import plot_components, plot_bmus, plot_umatrix

plot_components(model, feature_names)
plot_bmus(model)
plot_umatrix(model)
```

## Features
- **Lattices**: hexagonal, rectangular
- **Neighborhoods**: Gaussian, bubble, cut Gaussian, Epanechicov
- **Normalization**: standard, min-max, log, logistic, Box-Cox
- **Initialization**: random, PCA
- **Radius decay**: linear, exponential
- **Map size**: manual or auto-estimated from data
- **Metrics**: quantization error (MAE), topographic error, vacancy rate
- **Visualization**: component planes, U-matrix, BMU hit maps

## Examples

See the `examples/` folder for complete examples on real datasets:
- `toy/` — synthetic correlated data
- `spotify/` — Spotify audio features (114k tracks)
- `happiness/` — World Happiness Report (2015-2022)
- `creditcard/` — credit card fraud detection

## Autoresearch

The `autoresearch/` folder contains a setup for automated SOM hyperparameter optimization:
- `prepare.py` — downloads and processes all datasets (run once)
- `train.py` — trains SOMs on all datasets, prints metrics. This is the only file an agent modifies.

```bash
uv run python autoresearch/prepare.py
uv run python autoresearch/train.py
```

## Tests

```bash
uv run python -m pytest somnium/tests/ -v
```

## Attribution
This library was built using [SOMPY](https://github.com/sevamoo/SOMPY) as a starting point.

## License
MIT. See `LICENSE`. Copyright (c) 2019 Iván Vallés Pérez
