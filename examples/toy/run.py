
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from somnium.core import SOM
from somnium.visualization import plot_components, plot_bmus, plot_umatrix
from somnium.visualization import _calculate_bmus_matrix, _plot_bmus

PLOT=False

# %%
df = (pd.DataFrame({"weight": np.random.randn(10000)*8+75,
                    "height": np.random.randn(10000)*8+180})
      .assign(BMI=lambda d: 703*d.weight/(d.height**2),
              fat=lambda d: ((d.weight-75)/8)*2 + 23.5 +
              np.random.randn(10000)*3,
              random=10+np.random.randn(10000)
              ))

data = df.values[:-10] # We leave some data out for testing purposes
test_data = df.values[-100:] # Test data

names = df.columns

# %% [markdown]
# ## Hexagonal lattice SOM

# %%
model = SOM(lattice="hexa", normalization="standard", distance_metric="euclidean", neighborhood="gaussian", mapsize=[15, 11], n_jobs=1)

model.fit(data, 30, 20, 5)
model.fit(data, 30, 5, 1)

print("=== LATTICE HEX ===")
print("E_Quantization =", model.calculate_quantization_error())
print("E_Topographic =", model.calculate_topographic_error())

if PLOT:
        plot_components(model, names, figure_width=30, max_subplot_columns=3)
        plot_bmus(model, figure_width=10)
        plot_umatrix(model, colormap = plt.cm.hot, figure_width=10)

# %% [markdown]
# ### Inferring the test set's BMUs

# %%
test_set_bmus = model.predict(test_data)

# %%
if PLOT:
        bmu_hits=_calculate_bmus_matrix(test_set_bmus, model.codebook.n_rows, model.codebook.n_columns)
        fig=_plot_bmus(bmu_hits=bmu_hits, nrows=model.codebook.n_rows, ncols=model.codebook.n_columns, 
                lattice_name=model.codebook.lattice.name, 
                figure_width=10)
        ax=plt.gca()
        plt.show()

# %% [markdown]
# ## Rectangular lattice SOM

# %%
model = SOM(lattice="rect", normalization="standard", distance_metric="euclidean", neighborhood="epanechicov", mapsize=[15, 10])

model.fit(data, 30, 20, 5)
model.fit(data, 30, 5, 3)

print("=== LATTICE RECT ===")
print("E_Quantization =", model.calculate_quantization_error())
print("E_Topographic =", model.calculate_topographic_error())

if PLOT:
        plot_components(model=model, names=names, figure_width=30, max_subplot_columns=3)
        plot_bmus(model, figure_width=10)
        plot_umatrix(model=model, colormap = plt.cm.hot, figure_width=10)

# %%
test_set_bmus = model.predict(test_data)

# %% [markdown]
# ### Inferring the test set BMUs

# %%
if PLOT:
        bmu_hits=_calculate_bmus_matrix(test_set_bmus, model.codebook.n_rows, model.codebook.n_columns)
        fig=_plot_bmus(bmu_hits=bmu_hits, nrows=model.codebook.n_rows, 
                ncols=model.codebook.n_columns, lattice_name=model.codebook.lattice.name, 
                figure_width=10)
        ax=plt.gca()
        plt.show()


