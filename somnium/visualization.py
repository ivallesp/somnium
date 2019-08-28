import math
from collections import Counter
import logging
import itertools
import numpy as np
from scipy import signal
from matplotlib import cm, pyplot as plt
from matplotlib.collections import RegularPolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_hexa_grid_coordinates(x, y):
    coordinates = [x for row in -1 * np.array(list(range(x))) for x in
                   list(zip(np.arange(((row) % 2) * -0.5 + 1,  y + ((row) % 2) * -0.5 + 1), [np.sqrt(3) / 2 * (row)] * y))]
    return np.array(list(reversed(coordinates)))

def create_rect_grid_coordinates(x, y):
    coordinates = list(itertools.product(range(0, -x, -1), range(y)))
    return np.array(list(reversed(coordinates)))[:, ::-1]


def plot_map(d_matrix,
             titles=[],
             colormap=cm.gray,
             shape=[1, 1],
             comp_width=5,
             hex_shrink=1.0,
             lattice="hexa",
             mode="color",
             fig=None):
    """
    Plot hexagon map where each neuron is represented by a hexagon. The hexagon
    color is given by the distance between the neurons (D-Matrix)

    Args:
    - grid: Grid dictionary (keys: centers, x, y ),
    - d_matrix: array contaning the distances between each neuron
    - w: width of the map in inches
    - title: map title

    Returns the Matplotlib SubAxis instance
    """
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
    d_matrix = np.flip(d_matrix, axis=0)
    d_matrix = np.flip(d_matrix, axis=1)
    d_matrix = np.expand_dims(d_matrix, 2) if d_matrix.ndim < 3 else d_matrix
    titles = [""] * d_matrix.shape[2] if len(titles) != d_matrix.shape[2] else titles

    for comp, title in zip(range(d_matrix.shape[2]), titles):
        ax = fig.add_subplot(shape[0], shape[1], comp + 1, aspect='equal')
        dm = d_matrix[:, :, comp].reshape(np.multiply(*d_matrix.shape[:2]))
        ax, n_centers = plot_comp(dm=dm, title=title, ax=ax, map_shape=d_matrix.shape[:2], colormap=colormap,
                                  lattice=lattice, mode=mode)
    return ax, list(reversed(n_centers))


def plot_comp(dm, title, ax, map_shape, colormap, lattice="hexa", mode="color"):
    # discover radius and hexagon
    if lattice == "hexa":
        n_centers = create_hexa_grid_coordinates(*map_shape[:2])
        radius_f = lambda x: x * 2 / 3
        rotation = 0
        numsides = 6
    elif lattice == "rect":
        n_centers = create_rect_grid_coordinates(*map_shape[:2])
        radius_f = lambda x: x / np.sqrt(2)
        rotation = np.pi / 4
        numsides = 4

    # Get pixel size between two data points
    xpoints = n_centers[:, 0]
    ypoints = n_centers[:, 1]
    ax.scatter(xpoints, ypoints, s=0.0, marker='s')
    ax.axis([min(xpoints) - 1., max(xpoints) + 1.,
             min(ypoints) - 1., max(ypoints) + 1.])
    xy_pixels = ax.transData.transform(np.vstack([xpoints, ypoints]).T)
    xpix, ypix = xy_pixels.T
    radius = radius_f(abs(ypix[map_shape[1]] - ypix[0]))

    area_inner_circle = math.pi * (radius ** 2)

    if mode == "color":
        sizes = [area_inner_circle] * dm.shape[0]
    elif mode == "size":
        sizes = area_inner_circle * (dm.reshape(-1) / dm.max())
        dm = dm * 0

    collection_bg = RegularPolyCollection(
        numsides=numsides,  # a hexagon
        rotation=rotation,
        sizes=sizes,
        array=dm,
        cmap=colormap,
        offsets=n_centers,
        transOffset=ax.transData,
    )
    ax.add_collection(collection_bg, autolim=True)

    ax.axis('off')
    ax.autoscale_view()
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(collection_bg, cax=cax)
    if mode != "color":
        cbar.remove()
    return ax, n_centers


def plot_components(model, data, names, colormap=plt.cm.jet, max_subplot_columns=5, figure_width=20):
    codebook = model.normalizer.denormalize_by(data, model.codebook.matrix)
    codebook = codebook.reshape(list(model.codebook.mapsize) + [model.codebook.matrix.shape[-1]])
    subplot_cols = min(codebook.shape[-1], max_subplot_columns)
    subplot_rows = math.ceil(codebook.shape[-1] / subplot_cols)
    subplots_shape = [subplot_rows, subplot_cols]
    aspect_ratio = model.codebook.n_columns / model.codebook.n_rows
    xinch = figure_width
    comp_width = (figure_width / subplot_cols)
    yinch = comp_width * aspect_ratio * subplot_rows
    figsize = (xinch, yinch)
    fig = plt.figure(figsize=figsize, dpi=72)
    plot_map(codebook, titles=names, shape=subplots_shape, colormap=colormap, fig=fig,
             lattice=model.codebook.lattice.name, mode="color");
    plt.show()


def plot_bmus(model, figure_width=20):
    counts = Counter(model.bmu[0])
    counts = [counts.get(x, 0) for x in range(model.codebook.nnodes)]
    mp = np.array(counts).reshape(model.codebook.n_rows, model.codebook.n_columns)
    subplot_cols = 1
    subplot_rows = 1
    subplots_shape = (subplot_rows, subplot_cols)
    aspect_ratio = model.codebook.n_columns / model.codebook.n_rows
    xinch = figure_width
    comp_width = (figure_width / subplot_cols)
    yinch = comp_width * aspect_ratio * subplot_rows
    figsize = (xinch, yinch)
    fig = plt.figure(figsize=figsize, dpi=72.)
    plot_map(mp, shape=subplots_shape, colormap=plt.cm.winter, fig=fig, lattice=model.codebook.lattice.name,
             mode="size")
    plt.show()


def plot_umatrix(model, colormap=plt.cm.hot, figure_width=20):
    codebook = model.codebook.matrix.reshape(model.codebook.n_rows, model.codebook.n_columns, -1)
    if model.codebook.lattice.name == "rect":
        umat = calculate_umatrix_rect(codebook)
    elif model.codebook.lattice.name == "hexa":
        raise NotImplementedError
    subplot_cols = 1
    subplot_rows = 1
    subplots_shape = (subplot_rows, subplot_cols)
    aspect_ratio = umat.shape[1] / umat.shape[0]
    xinch = figure_width
    comp_width = (figure_width / subplot_cols)
    yinch = comp_width * aspect_ratio * subplot_rows
    figsize = (xinch, yinch)
    fig = plt.figure(figsize=figsize, dpi=72.)
    plot_map(umat, shape=subplots_shape, colormap=colormap, fig=fig, lattice=model.codebook.lattice.name,
             mode="color")
    plt.show()


def calculate_umatrix_rect(codebook):
    UMatrix = np.zeros((codebook.shape[0] + codebook.shape[0] - 1, codebook.shape[1] + codebook.shape[1] - 1))
    rows_dist = np.sqrt(((codebook[:-1, :, :] - codebook[1:, :, :]) ** 2).sum(axis=-1))  # Shift rows
    cols_dist = np.sqrt(((codebook[:, :-1, :] - codebook[:, 1:, :]) ** 2).sum(axis=-1))  # Shift columns
    UMatrix[1::2, ::2] = rows_dist  # Alternately insert
    UMatrix[::2, 1::2] = cols_dist  # Alternately insert
    kernel = np.array([[0, 1, 0],   # Kernel to fill the gaps with sum of neighbors
                       [1, 0, 1],
                       [0, 1, 0]])
    out = signal.convolve2d(UMatrix, kernel, boundary='fill', mode='same')
    out = out / signal.convolve2d(np.ones_like(UMatrix), kernel, boundary='fill', mode='same')
    UMatrix = UMatrix + out
    return UMatrix