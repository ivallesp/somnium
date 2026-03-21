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
    mapsize="auto",              # or (15, 10); auto estimates from data
    lattice="hexa",              # "hexa", "rect", "toroidal_hexa", "toroidal_rect",
                                 # "cylindrical_hexa", "cylindrical_rect"
    neighborhood="gaussian",     # "gaussian", "bubble", "cut_gaussian", "epanechicov"
    normalization="standard",    # "standard", "minmax", "log", "logistic", "boxcox"
    initialization="pca",        # "pca" or "random"
    distance_metric="euclidean", # any scipy distance metric
)

# Two-phase training (rough + fine), with optional exponential decay
model.fit(data, epochs=30, radiusin=20, radiusfin=5, decay="linear")
model.fit(data, epochs=30, radiusin=5,  radiusfin=1, decay="exponential")

# Or use the convenience method (auto radii from map size)
model.fit_auto(data)

# Metrics (lower is better for all)
model.calculate_quantization_error()  # per-feature MAE
model.calculate_topographic_error()   # fraction with non-adjacent BMUs
model.calculate_vacancy_rate()        # fraction of unused neurons

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
- **Lattices**: hexagonal, rectangular — flat, toroidal (both axes wrap), or cylindrical (columns wrap)
- **Neighborhoods**: Gaussian, bubble, cut Gaussian, Epanechicov
- **Normalization**: standard, min-max, log, logistic, Box-Cox
- **Initialization**: random, PCA
- **Training**: multi-phase with linear or exponential radius decay; `fit_auto()` convenience method
- **Map size**: manual or auto-estimated from data (5*sqrt(N) heuristic with PCA aspect ratio)
- **Metrics**: quantization error (MAE), topographic error, vacancy rate
- **Visualization**: component planes, U-matrix, BMU hit maps (hexagonal and rectangular)

## Examples

See the `examples/` folder for complete examples on real datasets:
- `toy/` — synthetic correlated data
- `spotify/` — Spotify audio features (114k tracks)
- `happiness/` — World Happiness Report (2015-2022)
- `creditcard/` — credit card fraud detection

## Tests

```bash
uv run python -m pytest somnium/tests/ -v
```

## Attribution
This library was built using [SOMPY](https://github.com/sevamoo/SOMPY) as a starting point.

## License
MIT. See `LICENSE`. Copyright (c) 2019-2026 Iván Vallés Pérez
