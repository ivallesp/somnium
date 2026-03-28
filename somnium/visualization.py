import math
from collections import Counter
import logging
import numpy as np
from scipy import signal
from matplotlib import cm, pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from somnium.lattice import LatticeFactory


def plot_components(model, names, colormap=plt.cm.jet, max_subplot_columns=5, figure_width=20):
    """
    High level function to plot the components of a Self-Organising Map model.
    :param model: model which codebook will be represented (somnia SOM model)
    :param names: names to be used for inserting the titles in the components of the figure (list or iterable)
    :param colormap: colormap to use to generate the plots (matplotlib.cm)
    :param max_subplot_columns: number of columns in the resulting figure subplots (int)
    :param figure_width: width of the figure (int)
    :return: None (void)
    """
    codebook = model.normalizer.denormalize(model.codebook.matrix)
    codebook = codebook.reshape(list(model.codebook.mapsize) + [model.codebook.matrix.shape[-1]])
    subplot_cols = min(codebook.shape[-1], max_subplot_columns)
    subplot_rows = math.ceil(codebook.shape[-1] / subplot_cols)
    subplots_shape = [subplot_rows, subplot_cols]
    aspect_ratio = model.codebook.n_rows / model.codebook.n_columns
    xinch = figure_width
    comp_width = (figure_width / subplot_cols)
    yinch = comp_width * aspect_ratio * subplot_rows
    figsize = (xinch, yinch)
    fig = plt.figure(figsize=figsize, dpi=72)
    plot_map(codebook, titles=names, shape=subplots_shape, colormap=colormap, fig=fig,
             lattice=model.codebook.lattice.name, mode="color");


def plot_bmus(model, figure_width=20):
    """
    High level function to plot the bmus hit-map of a Self-Organising Map model.
    :param model: model which codebook will be used to calculate and represent the plot (somnia SOM model)
    :param figure_width: width of the figure (int)
    :return: None (void)
    """
    bmu_hits = calculate_bmus_matrix(model)
    nrows = model.codebook.n_rows
    ncols = model.codebook.n_columns
    lattice=model.codebook.lattice.name
    return _plot_bmus(bmu_hits=bmu_hits, nrows=nrows, ncols=ncols, lattice_name=lattice, figure_width=figure_width)


def _plot_bmus(bmu_hits, nrows, ncols, lattice_name, figure_width=20):

    subplot_cols = 1
    subplot_rows = 1
    subplots_shape = (subplot_rows, subplot_cols)
    aspect_ratio = nrows / ncols
    xinch = figure_width
    comp_width = (figure_width / subplot_cols)
    yinch = comp_width * aspect_ratio * subplot_rows
    figsize = (xinch, yinch)
    fig = plt.figure(figsize=figsize, dpi=72.)
    plot_map(bmu_hits, shape=subplots_shape, colormap=plt.cm.winter, fig=fig, lattice=lattice_name,
             mode="size")
    return fig


def plot_umatrix(model, colormap=plt.cm.hot, figure_width=20):
    """
    High level function to plot the U-Matrix of a Self-Organising Map codebook.
    :param model: model which codebook will be used to calculate and represent the U-Matrix (somnia SOM model)
    :param data: original dataset used to denormalize the codebook (np.array)
    :param colormap: colormap to use to generate the plots (matplotlib.cm)
    :param figure_width: width of the figure (int)
    :return: None (void)
    """
    codebook = model.codebook.matrix.reshape(model.codebook.n_rows, model.codebook.n_columns, -1)
    if model.codebook.lattice.name.endswith("rect"):
        umat = calculate_umatrix_rect(codebook)
    elif model.codebook.lattice.name.endswith("hexa"):
        umat = calculate_umatrix_hexa(codebook)
    subplot_cols = 1
    subplot_rows = 1
    subplots_shape = (subplot_rows, subplot_cols)
    aspect_ratio = umat.shape[0] / umat.shape[1]
    xinch = figure_width
    comp_width = (figure_width / subplot_cols)
    yinch = comp_width * aspect_ratio * subplot_rows
    figsize = (xinch, yinch)
    fig = plt.figure(figsize=figsize, dpi=72.)
    plot_map(umat, shape=subplots_shape, colormap=colormap, fig=fig, lattice=model.codebook.lattice.name,
             mode="color")


def plot_map(components_matrix,
             titles=tuple(),
             colormap=cm.gray,
             shape=(1, 1),
             lattice="hexa",
             mode="color",
             fig=None):
    """
    Plots a map of components. It works for hexagonal and rectangular lattice as well as with color and element size
    representations.
    :param components_matrix: matrix containing all the components to be plotted in the map. The shape must be:
    (n_rows, n_cols, n_components). The resulting map will have as many components as the last axis of the
    matrix provided (n_components). (np.array)
    :param titles: list containing the names of the components to be plotted. If not provided, no titles will be
    added to the components in the resulting figure (list or iterable)
    :param colormap: colormap to use to generate the plots (matplotlib.cm)
    :param shape: shape for the componentds subplots. It should have the following format: (n_rows, n_columns).
    In order to be valid, n_rows * columns >= n_components (tuple)
    :param lattice: lattice to be used to plot the components. Allowed lattices are 'rect' and 'hexa', for rectangular
    and hexagonal grids (str)
    :param mode: indicates how the data should be represented in the components (str). There are two possibilities:
      - 'color': the values of the components will be represented as colors in the grid
      - 'size': the value of the components will be represented as sizes of the elements of the grid.
    :param fig: figure to use to plot the components map (matplotlib.pyplot.figure)
    :return: axis of the last component and list of coordinates of the centers (tuple)
    """
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
    components_matrix = np.flip(components_matrix, axis=0)
    components_matrix = np.flip(components_matrix, axis=1)
    components_matrix = np.expand_dims(components_matrix, 2) if components_matrix.ndim < 3 else components_matrix
    titles = [""] * components_matrix.shape[2] if len(titles) != components_matrix.shape[2] else titles

    for comp_idx, title in zip(range(components_matrix.shape[2]), titles):
        ax = fig.add_subplot(shape[0], shape[1], comp_idx + 1, aspect='equal')
        component_matrix = components_matrix[:, :, comp_idx].reshape(np.multiply(*components_matrix.shape[:2]))
        ax, coordinates = plot_comp(component_matrix=component_matrix, title=title, ax=ax,
                                    map_shape=components_matrix.shape[:2], colormap=colormap,lattice=lattice, mode=mode)
    return ax, list(reversed(coordinates))


def plot_comp(component_matrix, title, ax, map_shape, colormap, lattice="hexa", mode="color"):
    """
    Plots a component into a rectangular or hexagonal lattice. It allows 'color' and 'size' modes, meaning that the
    data in the component matrix can be represented as colors or as sizes of the elements in the plot
    :param component_matrix: matrix containing the data which is intended to be ploted. It must be a 2-D matrix
    (np.array)
    :param title: title to be assigned to the component (str)
    :param ax: matplotlib axis to use to plot the component (matplotlib axis)
    :param map_shape: shape of the codebook matrices (tuple)
    :param colormap: colormap to use to generate the plots (matplotlib.cm)
    :param lattice: lattice to be used to plot the components. Allowed lattices are 'rect' and 'hexa', for rectangular
    and hexagonal grids (str)
    :param mode: indicates how the data should be represented in the components (str). There are two possibilities:
      - 'color': the values of the components will be represented as colors in the grid
      - 'size': the value of the components will be represented as sizes of the elements of the grid.
    :return: axis of the last component and list of coordinates of the centers (tuple)
    """
    # describe rectangle or hexagon
    if lattice.endswith("hexa"):
        numsides = 6
        rotation = 0
    elif lattice.endswith("rect"):
        numsides = 4
        rotation = np.pi / 4

    coordinates = LatticeFactory.build(lattice).generate_lattice(*map_shape[:2])
    # Sort to draw left-right top-bottom
    coordinates = coordinates.copy()
    coordinates = coordinates[:, ::-1]
    coordinates = coordinates[np.lexsort([-coordinates[:, 0], -coordinates[:, 1]])]
    coordinates[:, 1] = -coordinates[:, 1]

    # Compute radius in data coordinates from spacing between adjacent rows
    xpoints = coordinates[:, 0]
    ypoints = coordinates[:, 1]
    row_spacing = abs(ypoints[map_shape[1]] - ypoints[0])
    if lattice.endswith("hexa"):
        radius = row_spacing * 2 / 3
    elif lattice.endswith("rect"):
        radius = row_spacing / np.sqrt(2)

    if mode == "size":
        scale = component_matrix.reshape(-1) / component_matrix.max()
        component_matrix = component_matrix * 0
    else:
        scale = np.ones(component_matrix.shape[0])

    patches = []
    for i, (xy, s) in enumerate(zip(coordinates, scale)):
        patches.append(RegularPolygon(xy, numVertices=numsides, radius=radius * np.sqrt(s),
                                      orientation=rotation))
    collection_bg = PatchCollection(patches, cmap=colormap, match_original=False)
    collection_bg.set_array(component_matrix)
    ax.add_collection(collection_bg)

    ax.set_xlim(min(xpoints) - 1., max(xpoints) + 1.)
    ax.set_ylim(min(ypoints) - 1., max(ypoints) + 1.)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(collection_bg, cax=cax)
    if mode != "color":
        cbar.remove()
    return ax, coordinates


def plot_label_map(model, labels, colormap=plt.cm.YlOrRd, max_subplot_columns=5, figure_width=20):
    """
    Plots per-class proportion heatmaps on the SOM. One subplot per unique label showing,
    for each neuron, the fraction of its assigned data points that belong to that class.
    :param model: trained SOM model
    :param labels: array-like of labels, one per training data point (same order as training data)
    :param colormap: colormap for the proportion heatmaps (matplotlib colormap)
    :param max_subplot_columns: max number of subplot columns (int)
    :param figure_width: width of the figure (int)
    :return: figure (matplotlib.figure.Figure)
    """
    labels = np.asarray(labels)
    bmu_indices = model.bmu[0].astype(int)
    unique_labels = sorted(np.unique(labels).tolist())
    n_rows = model.codebook.n_rows
    n_cols = model.codebook.n_columns
    nnodes = model.codebook.nnodes

    # Compute per-neuron proportion for each label
    proportions = np.zeros((n_rows, n_cols, len(unique_labels)))
    hits = np.zeros(nnodes)
    for node in range(nnodes):
        mask = bmu_indices == node
        hits[node] = mask.sum()
        if hits[node] > 0:
            node_labels = labels[mask]
            for k, lab in enumerate(unique_labels):
                proportions[node // n_cols, node % n_cols, k] = (node_labels == lab).sum() / hits[node]

    subplot_cols = min(len(unique_labels), max_subplot_columns)
    subplot_rows = math.ceil(len(unique_labels) / subplot_cols)
    subplots_shape = [subplot_rows, subplot_cols]
    aspect_ratio = n_rows / n_cols
    comp_width = figure_width / subplot_cols
    figsize = (figure_width, comp_width * aspect_ratio * subplot_rows)
    fig = plt.figure(figsize=figsize, dpi=72)
    titles = [str(lab) for lab in unique_labels]
    plot_map(proportions, titles=titles, shape=subplots_shape, colormap=colormap,
             fig=fig, lattice=model.codebook.lattice.name, mode="color")
    return fig


def calculate_bmus_matrix(model):
    """
    Function used to calculate how many times each neuron has been a best matching unit
    :param model: model which codebook will be used to calculate and represent the plot (somnia SOM model)
    :return: bmu_hits matrix (np.array)
    """
    bmu_list = model.bmu[0]
    nrows = model.codebook.n_rows
    ncols = model.codebook.n_columns
    return _calculate_bmus_matrix(bmu_list, nrows, ncols)


def _calculate_bmus_matrix(bmu_list, nrows, ncols):
    nnodes = nrows*ncols
    counts = Counter(bmu_list)
    counts = [counts.get(x, 0) for x in range(nnodes)]
    bmu_hits = np.array(counts).reshape(nrows, ncols)
    return bmu_hits


def calculate_umatrix_rect(codebook):
    """
    Calculates the U-Matrix of a codebook with a rectangular lattice
    :param codebook: codebook to be used to calculate the U-Matrix (np.array)
    :return: U-Matrix (np.array)
    """
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
    mask = (UMatrix == 0)
    UMatrix[mask] = out[mask]
    return UMatrix


def calculate_umatrix_hexa(codebook):
    """
    Calculates the U-Matrix of a codebook with a hexagonal lattice
    :param codebook: codebook to be used to calculate the U-Matrix (np.array)
    :return: U-Matrix (np.array)
    """
    UMatrix = np.zeros((codebook.shape[0] + codebook.shape[0] - 1, codebook.shape[1] + codebook.shape[1]))
    # Track which cells have been filled with valid distance data
    filled = np.zeros_like(UMatrix, dtype=bool)

    # Diagonal lefts
    diag_left_odd = np.sqrt(
        ((codebook[::2, ][(((codebook.shape[0] % 2 != 0) + 0)):] - codebook[1::2, ]) ** 2).sum(axis=-1))
    diag_left_even = np.sqrt(((codebook[1:-1:2,1:] - codebook[2::2,:-1])**2).sum(axis=-1))
    diag_left_even = np.column_stack([np.zeros([diag_left_even.shape[0], 1]), diag_left_even])
    UMatrix[3::4,::2] = diag_left_even
    filled[3::4,::2] = True
    UMatrix[1::4,1::2] = diag_left_odd
    filled[1::4,1::2] = True

    # Diagonal right
    diag_right_odd = np.sqrt(
        ((codebook[::2, :-1][(((codebook.shape[0] % 2 != 0) + 0)):] - codebook[1::2, 1:]) ** 2).sum(axis=-1))
    diag_right_odd = np.column_stack([np.zeros([diag_right_odd.shape[0], 1]), diag_right_odd])
    diag_right_even = np.sqrt(((codebook[1:-1:2] - codebook[2::2])**2).sum(axis=-1))
    UMatrix[1::4,::2] = diag_right_odd
    filled[1::4,::2] = True
    UMatrix[3::4,1::2] = diag_right_even
    filled[3::4,1::2] = True

    # Sides
    sides =  np.sqrt(((codebook[:,1:] - codebook[:,:-1])**2).sum(axis=-1))
    sides = np.array([[0.0] + x.tolist() if (i%2 == 0) else x.tolist()+[0.0] for i,x in enumerate(sides)])
    UMatrix[::4,::2] = sides[::2]
    filled[::4,::2] = True
    UMatrix[2::4,1::2] = sides[1::2]
    filled[2::4,1::2] = True

    # Odds conv
    kernel = np.array([[1, 1, 0],   # Kernel to fill the gaps with sum of neighbors
                       [1, 0, 1],
                       [1, 1, 0]])
    out = signal.convolve2d(UMatrix, kernel, boundary='fill', mode='same')
    count_mat = filled.astype(float)
    denom = signal.convolve2d(count_mat, kernel, boundary='fill', mode='same')
    denom[denom == 0] = 1
    out = out / denom
    UMatrix[::4, 1::2] = out[::4, 1::2]
    filled[::4, 1::2] = True

    # Evens conv
    kernel = np.array([[0, 1, 1],   # Kernel to fill the gaps with sum of neighbors
                       [1, 0, 1],
                       [0, 1, 1]])
    out = signal.convolve2d(UMatrix, kernel, boundary='fill', mode='same')
    count_mat = filled.astype(float)
    denom = signal.convolve2d(count_mat, kernel, boundary='fill', mode='same')
    denom[denom == 0] = 1
    out = out / denom
    UMatrix[2::4, ::2] = out[2::4, ::2]
    filled[2::4, ::2] = True

    # Filling the voids #0
    kernel = np.array([[0, 1, 1],   # Kernel to fill the gaps with sum of neighbors
                       [1, 0, 1],
                       [0, 1, 1]])
    out = signal.convolve2d(UMatrix, kernel, boundary='fill', mode='same')
    count_mat = filled.astype(float)
    denom = signal.convolve2d(count_mat, kernel, boundary='fill', mode='same')
    denom[denom == 0] = 1
    out = out / denom
    UMatrix[::2, 0][::2] = out[::2, 0][::2]
    UMatrix[2::4,-1] = out[2::4,-1]

    # Filling the voids #1
    kernel = np.array([[1, 1, 0],   # Kernel to fill the gaps with sum of neighbors
                       [1, 0, 1],
                       [1, 1, 0]])
    out = signal.convolve2d(UMatrix, kernel, boundary='fill', mode='same')
    count_mat = filled.astype(float)
    denom = signal.convolve2d(count_mat, kernel, boundary='fill', mode='same')
    denom[denom == 0] = 1
    out = out / denom
    UMatrix[1::2, 0] = out[1::2, 0]
    return UMatrix