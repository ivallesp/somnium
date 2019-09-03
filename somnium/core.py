import itertools

import numpy as np
import scipy as sp

from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix

from somnium.codebook import Codebook
from somnium.normalization import NormalizerFactory
from somnium.neighborhood import NeighborhoodFactory


def batching(list_of_iterables, n=1, infinite=False, return_incomplete_batches=False):
    list_of_iterables = [list_of_iterables] if type(list_of_iterables) is not list else list_of_iterables
    assert (len({len(it) for it in list_of_iterables}) == 1)
    length = len(list_of_iterables[0])
    while 1:
        for ndx in range(0, length, n):
            if not return_incomplete_batches:
                if (ndx + n) > length:
                    break
            yield [iterable[ndx:min(ndx + n, length)] for iterable in list_of_iterables]

        if not infinite:
            break


def find_bmu(codebook, input_matrix, metric, njb=1, nth=1):
    """
    Finds the best matching unit (bmu) for each input data from the input
    matrix. It does all at once parallelizing the calculation instead of
    going through each input and running it against the codebook.

    :param input_matrix: numpy matrix representing inputs as rows and
        features/dimension as cols
    :param njb: number of jobs to parallelize the search
    :returns: the best matching unit for each input
    """
    dlen = input_matrix.shape[0]
    njb = cpu_count() if njb == -1 else njb
    chunks = list(batching([input_matrix], n=dlen // njb, return_incomplete_batches=True))

    with Pool(njb) as pool:
        bmus = pool.map(lambda chk: _chunk_based_bmu_find(input_matrix=chk,
                                                          codebook=codebook.matrix,
                                                          metric=metric,
                                                          nth=nth), chunks)
    bmus = np.asarray(list(itertools.chain(*bmus))).T
    return bmus


def update_codebook_voronoi(codebook, training_data, bmu, neighborhood, _dlen):
    """
    Updates the weights of each node in the codebook that belongs to the
    bmu's neighborhood.

    First finds the Voronoi set of each node. It needs to calculate a
    smaller matrix.
    Super fast comparing to classic batch training algorithm, it is based
    on the implemented algorithm in som toolbox for Matlab by Helsinky
    University.

    :param training_data: input matrix with input vectors as rows and
        vector features as cols
    :param bmu: best matching unit for each input data. Has shape of
        (2, dlen) where first row has bmu indexes
    :param neighborhood: matrix representing the neighborhood of each bmu

    :returns: An updated codebook that incorporates the learnings from the
        input data
    """
    row = bmu[0].astype(int)
    col = np.arange(_dlen)
    val = np.tile(1, _dlen)
    p = csr_matrix((val, (row, col)),
                   shape=(codebook.nnodes, _dlen))  # Matrix (nnodes, dlen) indicating the BMU for each instance (OHE)
    s = p.dot(
        training_data)  # (nnodes, data_length) x (data_length, feats) = (nnodes, feats). Sum data for all BMUs, we will divide to compute average later

    # neighborhood has nnodes*nnodes and S has nnodes*dim
    # ---> Nominator has nnodes*dim
    # I am not sure if I have to transpose the neighborhood
    numerator = neighborhood.T.dot(s)  # Weight by neighborhood function, adds neighbors value
    nV = p.sum(axis=1).reshape(1, codebook.nnodes)  # Number of hits per BMU
    denominator = nV.dot(neighborhood.T).reshape(codebook.nnodes,
                                                 1)  # Weight by neighborhood function (allows dividing apples by apples)
    new_codebook = np.divide(numerator, denominator)  # Compute the average
    return np.around(new_codebook, decimals=6)


def _chunk_based_bmu_find(input_matrix, codebook, metric, nth=1):
    assert len(input_matrix) == 1
    input_matrix = input_matrix[0]
    dlen = input_matrix.shape[0]
    bmu = np.empty((dlen, 2))
    d = sp.spatial.distance.cdist(codebook, input_matrix, metric=metric)
    bmu[:, 0] = np.argpartition(d, nth, axis=0)[nth - 1]  # BMU index
    bmu[:, 1] = np.partition(d, nth, axis=0)[nth - 1]  # Distance (without the X contribution)
    return bmu


def bmu_ind_to_xy(bmu_ind, codebook):
    """
    Translates a best matching unit index to the corresponding matrix x,y coordinates for the rectangular lattice

    :param bmu_ind: node index of the best matching unit
        (number of node from top left node)
    :returns: corresponding (x,y) coordinate
    """
    rows = codebook.mapsize[0]
    cols = codebook.mapsize[1]

    # bmu should be an integer between 0 to no_nodes
    out = np.zeros((bmu_ind.shape[0], 3))
    out[:, 2] = bmu_ind
    out[:, 0] = rows - 1 - bmu_ind / cols
    out[:, 0] = bmu_ind / cols
    out[:, 1] = bmu_ind % cols

    return out.astype(int)


def train_som(data, codebook, epochs, radiusin, radiusfin, neighborhood_f, distance_matrix, distance_metric, n_jobs):
    radius = np.linspace(radiusin, radiusfin, epochs)

    for i in range(epochs):
        neighborhood = neighborhood_f.calculate(distance_matrix, radius[i], codebook.nnodes)
        bmu = find_bmu(codebook, data, metric=distance_metric, njb=n_jobs)
        codebook.matrix = update_codebook_voronoi(codebook, data, bmu, neighborhood, data.shape[0])

    return bmu


class SOM:
    def __init__(self, neighborhood="gaussian", normalization="standard", mapsize=(15, 10), lattice="hexa",
                 distance_metric="euclidean", n_jobs=1):
        """
        :param neighborhood: neighborhood function to use in the planar lattice. Supported functions are: 'gaussian'
        (default), 'bubble', 'cut_gaussian' and 'epanechicov'. (str)
        :param normalization: technique to use for normalization. Supported methods are: 'standard' (default), 'minmax',
        'boxcox', 'logistic'. (str)
        :param mapsize: size of the component matrices (tuple or iterable)
        :param lattice: type of lattice: 'rect' or 'hexa' for rectangular and hexagonal lattices (str).
        :param distance_metric: distance metric to be used in all the operations (str). The supported distance metrics
        are the same ones as the supported by scipy: 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'euclidean', 'jensenshannon', 'mahalanobis', 'minkowski', 'seuclidean', 'sqeuclidean' and
        'wminkowski'. More information here: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
        :param n_jobs: number of jobs to use for run the algorithm. Parallelization done when finding the BMUs, using
        multiprocessing library.
        """
        self.neighborhood_calculator = NeighborhoodFactory.build(neighborhood)
        self.normalizer = NormalizerFactory.build(normalization)
        self.codebook = Codebook(mapsize=mapsize, lattice=lattice, distance_metric=distance_metric)
        self.distance_matrix = self.codebook.lattice.distances.reshape(self.codebook.nnodes, self.codebook.nnodes)
        self.n_jobs = n_jobs
        self.model_is_unfitted = True
        self.bmu = None

    def fit(self, data, epochs, radiusin, radiusfin):
        data_norm = self.normalizer.normalize(data)
        self.data_norm = data_norm
        _dlen = data_norm.shape[0]
        if self.model_is_unfitted:
            self.codebook.random_initialization(data_norm)
        self.bmu = train_som(data_norm, self.codebook, epochs, radiusin, radiusfin,
                             neighborhood_f=self.neighborhood_calculator,
                             distance_matrix=self.distance_matrix,
                             distance_metric=self.codebook.lattice.distance_metric,
                             n_jobs=self.n_jobs)
        return self

    def calculate_quantization_error(self):
        neuron_values = self.codebook.matrix[find_bmu(self.codebook, self.data_norm,
                                                      metric=self.codebook.lattice.distance_metric)[0].astype(int)]
        quantization_error = np.mean(np.abs(neuron_values - self.data_norm))
        return quantization_error

    def calculate_topographic_error(self, n_jobs=1):
        bmus1 = \
        find_bmu(self.codebook, self.data_norm, metric=self.codebook.lattice.distance_metric, njb=n_jobs, nth=1)[0]
        bmus2 = \
        find_bmu(self.codebook, self.data_norm, metric=self.codebook.lattice.distance_metric, njb=n_jobs, nth=2)[0]
        neighs = [self.codebook.lattice.are_neighbor_indices(int(x1), int(x2)) for (x1, x2) in zip(bmus1, bmus2)]
        return (1 - np.mean(neighs))

