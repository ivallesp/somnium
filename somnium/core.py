import itertools

import numpy as np

from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from somnium.codebook import Codebook
from somnium.normalization import NormalizerFactory
from somnium.neighborhood import NeighborhoodFactory
from somnium.util import batching


class SOM:
    def __init__(self, neighborhood="gaussian", normalization="standard", mapsize=(15, 10), lattice="hexa",
                 distance_metric="euclidean", n_jobs=1):
        """
         Principal somnia class, in charge of training the Self Organizing Map.
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
        self.data_norm = None

    def fit(self, data, epochs, radiusin, radiusfin):
        """
        Runs a set of epochs with the specified parameters and returns the model trained. The this method can be run
        several times in order to perform 'rough' and 'finetune' stages.
        :param data: dataset to use to train the model (np.array)
        :param epochs: number of epochs to run (int)
        :param radiusin: size of the radius of the neighborhood function at the first epoch (float)
        :param radiusfin: size of the radius of the neighborhood function at the last epoch (float)
        :return: the model trained (SOM)
        """
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
        """
        Calculates the quantization error of the model, i.e. the error which is originated by assigning the instances
        to the BMUs. It is calculated as the average Mean Absolute Error (MAE) between the instances and their assigned
        BMU.
        :return: the quantization error (float)
        """
        neuron_values = self.codebook.matrix[find_bmu(self.codebook, self.data_norm,
                                                      metric=self.codebook.lattice.distance_metric)[0].astype(int)]
        quantization_error = np.mean(np.abs(neuron_values - self.data_norm))
        return quantization_error

    def calculate_topographic_error(self, n_jobs=1):
        """
        Calculates the topographic error of the model, i.e. the percentage of instances for which the BMU and the
        2nd BMU are not immediate neighbors.
        :param n_jobs: number of jobs used to find the bmus (int)
        :return: the topographic error (float)
        """
        metric = self.codebook.lattice.distance_metric
        bmus1 = find_bmu(self.codebook, self.data_norm, metric=metric, njb=n_jobs, nth=1)[0]
        bmus2 = find_bmu(self.codebook, self.data_norm, metric=metric, njb=n_jobs, nth=2)[0]
        neighs = [self.codebook.lattice.are_neighbor_indices(int(x1), int(x2)) for (x1, x2) in zip(bmus1, bmus2)]
        return 1 - np.mean(neighs)


def train_som(data, codebook, epochs, radiusin, radiusfin, neighborhood_f, distance_matrix, distance_metric, n_jobs):
    """
    Takes a model and a data set as input and trains the model, given a set of parameters.
    :param data: input data to train the model (np.array)
    :param codebook: codebook of the model (np.array)
    :param epochs: number of epochs to train the model on the specified data (int)
    :param radiusin: initial radius of the neighborhood function, on the first epoch (float)
    :param radiusfin: final radius of the neighborhood function, on the last epoch (float)
    :param neighborhood_f: neighborhood function for establishing a relationship between each element in the lattice and
    its neighbors (Neighborhood class)
    :param distance_matrix: matrix of distances between the instances (np.array)
    :param distance_metric: name of the distance metric (str)
    :param n_jobs: number of jobs to use to train the model (int)
    :return: bmus for each data instance (np.array
    """
    radius = np.linspace(radiusin, radiusfin, epochs)

    for i in range(epochs):
        neighborhood = neighborhood_f.calculate(distance_matrix, radius[i], codebook.nnodes)
        bmu = find_bmu(codebook, data, metric=distance_metric, njb=n_jobs)
        codebook.matrix = update_codebook_voronoi(codebook, data, bmu, neighborhood, data.shape[0])
    return bmu


def update_codebook_voronoi(codebook, training_data, bmu, neighborhood, _dlen):
    """
    Updates the weights of each node in the codebook that belongs to the
    bmu's neighborhood.

    First finds the Voronoi set of each node. It needs to calculate a
    smaller matrix.
    Super fast comparing to classic batch training algorithm, it is based
    on the implemented algorithm in som toolbox for Matlab by Helsinky
    University.
    :param codebook: codebook object to be updated (somnium.codebook)
    :param training_data: input matrix with input vectors as rows and
        vector features as cols
    :param bmu: best matching unit for each input data. Has shape of
        (2, dlen) where first row has bmu indexes
    :param neighborhood: matrix representing the neighborhood of each bmu
    :param _dlen: number of instances in the data (int)
    :return: An updated codebook that incorporates the learnings from the
        input data
    """
    row = bmu[0].astype(int)
    col = np.arange(_dlen)
    val = np.tile(1, _dlen)
    # Matrix (nnodes, dlen) indicating the BMU for each instance (OHE)
    p = csr_matrix((val, (row, col)), shape=(codebook.nnodes, _dlen))
    # (nnodes, data_length) x (data_length, feats) = (nnodes, feats). Sum data for all BMUs, we will divide to
    # compute average later
    s = p.dot(training_data)

    # neighborhood has nnodes*nnodes and S has nnodes*dim
    # ---> Nominator has nnodes*dim
    # Weight by neighborhood function, adds neighbors value
    numerator = neighborhood.T.dot(s)
    # Number of hits per BMU
    n_v = p.sum(axis=1).reshape(1, codebook.nnodes)
    # Weight by neighborhood function (allows dividing apples by apples)
    denominator = n_v.dot(neighborhood.T).reshape(codebook.nnodes, 1)
    new_codebook = np.divide(numerator, denominator)  # Compute the average
    return np.around(new_codebook, decimals=6)


def _chunk_based_bmu_find(instance, codebook, distance_metric, nth=1):
    """
    Function in charge of finding the nth BMU of a given instance. This function is prepared to be parallelized.
    :param instance: instance for which the BMU is going to be determined (np.array)
    :param codebook: current codebook of the model (somnium.codebook.Codebook)
    :param distance_metric: distance metric to use for determining the BMU (str)
    :param nth: rank of the best matching unit to look for. 1st by default. 2nd is used for calculating the
    topographic error (int)
    :return: bmu index (int)
    """
    assert len(instance) == 1
    instance = instance[0]
    dlen = instance.shape[0]
    bmu = np.empty((dlen, 2))
    d = cdist(codebook, instance, metric=distance_metric)
    bmu[:, 0] = np.argpartition(d, nth, axis=0)[nth - 1]  # BMU index
    bmu[:, 1] = np.partition(d, nth, axis=0)[nth - 1]  # Distance (without the X contribution)
    return bmu


def find_bmu(codebook, input_matrix, metric, njb=1, nth=1):
    """
    Finds the best matching unit (bmu) for each input observation from the input matrix. It does all at once
    parallelizing the calculation instead of going through each input and running it against the codebook.

    :param codebook: codebook matrix to compare the input matrix with (np.array)
    :param input_matrix: numpy matrix representing inputs as rows and features/dimension as cols (np.array)
    :param metric: name of the distance metric to use. See scipy cdist/pdist documentation for more info (str)
    :param njb: number of jobs to parallelize the search (int)
    :param nth: rank of the best matching unit to look for. 1st by default. 2nd is used for calculating the
    topographic error (int)
    :returns: the best matching unit for each input (list of ints)
    """
    dlen = input_matrix.shape[0]
    njb = cpu_count() if njb == -1 else njb
    chunks = list(batching([input_matrix], n=dlen // njb, return_incomplete_batches=True))

    with Pool(njb) as pool:
        bmus = pool.map(lambda chk: _chunk_based_bmu_find(instance=chk,
                                                          codebook=codebook.matrix,
                                                          distance_metric=metric,
                                                          nth=nth), chunks)
    bmus = np.asarray(list(itertools.chain(*bmus))).T
    return bmus
