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
from somnium.exceptions import ModelNotTrainedError, InvalidValuesInDataSet


def estimate_mapsize(data):
    """
    Estimates a good map size for the given data using a heuristic: total neurons ≈ 5 * sqrt(N),
    aspect ratio derived from the ratio of the first two PCA eigenvalues.
    :param data: input data matrix (np.array)
    :return: (n_rows, n_columns) tuple
    """
    from sklearn.decomposition import PCA
    n_samples = data.shape[0]
    total_neurons = int(5 * np.sqrt(n_samples))
    if data.shape[1] >= 2:
        pca = PCA(n_components=2, svd_solver='randomized')
        pca.fit(data)
        ratio = pca.explained_variance_[0] / max(pca.explained_variance_[1], 1e-10)
        aspect = np.sqrt(ratio)
    else:
        aspect = 1.0
    n_cols = max(1, int(np.sqrt(total_neurons * aspect)))
    n_rows = max(1, int(total_neurons / n_cols))
    return (n_rows, n_cols)


class SOM:
    def __init__(self, neighborhood="gaussian", normalization="standard", mapsize=(15, 10), lattice="hexa",
                 distance_metric="euclidean", n_jobs=1, initialization="random"):
        """
         Principal somnium class, in charge of training the Self Organizing Map.
        :param neighborhood: neighborhood function to use in the planar lattice. Supported functions are: 'gaussian'
        (default), 'bubble', 'cut_gaussian' and 'epanechicov'. (str)
        :param normalization: technique to use for normalization. Supported methods are: 'standard' (default), 'minmax',
        'boxcox', 'logistic'. (str)
        :param mapsize: size of the component matrices (tuple or iterable), or 'auto' to estimate from data on first fit
        :param lattice: type of lattice: 'rect' or 'hexa' for rectangular and hexagonal lattices (str).
        :param distance_metric: distance metric to be used in all the operations (str). The supported distance metrics
        are the same ones as the supported by scipy: 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'euclidean', 'jensenshannon', 'mahalanobis', 'minkowski', 'seuclidean', 'sqeuclidean' and
        'wminkowski'. More information here: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
        :param n_jobs: number of jobs to use for run the algorithm. Parallelization done when finding the BMUs, using
        multiprocessing library.
        :param initialization: codebook initialization method. 'random' (default) or 'pca'. (str)
        """
        if initialization not in ("random", "pca"):
            raise ValueError(f"Unknown initialization method '{initialization}'. Use 'random' or 'pca'.")
        self.neighborhood_calculator = NeighborhoodFactory.build(neighborhood)
        self.normalizer = NormalizerFactory.build(normalization)
        self._mapsize = mapsize
        self._lattice = lattice
        self._distance_metric = distance_metric
        if mapsize != "auto":
            self._init_codebook(mapsize, lattice, distance_metric)
        self.n_jobs = n_jobs
        self.metric = distance_metric
        self.initialization = initialization
        self.model_is_unfitted = True
        self.bmu = None
        self.data_norm = None

    def _init_codebook(self, mapsize, lattice, distance_metric):
        self.codebook = Codebook(mapsize=mapsize, lattice=lattice, distance_metric=distance_metric)
        self.distance_matrix = self.codebook.lattice.distances.reshape(self.codebook.nnodes, self.codebook.nnodes)

    def fit(self, data, epochs, radiusin, radiusfin, decay="linear",
            learning_rate=1.0, subsample_ratio=1.0):
        """
        Runs a set of epochs with the specified parameters and returns the model trained. The this method can be run
        several times in order to perform 'rough' and 'finetune' stages.
        :param data: dataset to use to train the model (np.array)
        :param epochs: number of epochs to run (int)
        :param radiusin: size of the radius of the neighborhood function at the first epoch (float)
        :param radiusfin: size of the radius of the neighborhood function at the last epoch (float)
        :param decay: radius decay schedule. 'linear' (default) or 'exponential'. (str)
        :param learning_rate: blending factor for codebook update. 1.0 (default) = full batch
        update. Values < 1.0 blend old and new codebook for smoother updates. (float)
        :param subsample_ratio: fraction of data to use each epoch. 1.0 (default) = all data.
        Values < 1.0 randomly subsample each epoch for stochastic training. (float)
        :return: the model trained (SOM)
        """
        data = _check_data(data)
        data_norm = self.normalizer.normalize(data)
        self.data_norm = data_norm
        _dlen = data_norm.shape[0]
        if self.model_is_unfitted:
            if self._mapsize == "auto":
                self._init_codebook(estimate_mapsize(data_norm), self._lattice, self._distance_metric)
            if self.initialization == "pca":
                self.codebook.pca_linear_initialization(data_norm)
            else:
                self.codebook.random_initialization(data_norm)
            self.model_is_unfitted = False
        self.bmu = train_som(data_norm, self.codebook, epochs, radiusin, radiusfin,
                             neighborhood_f=self.neighborhood_calculator,
                             distance_matrix=self.distance_matrix,
                             distance_metric=self.codebook.lattice.distance_metric,
                             n_jobs=self.n_jobs, decay=decay,
                             learning_rate=learning_rate,
                             subsample_ratio=subsample_ratio)
        # Update the bmus at the end
        self.bmu = find_bmu(self.codebook,
                            data_norm,
                            metric=self.codebook.lattice.distance_metric,
                            njb=self.n_jobs)

        return self

    def fit_auto(self, data, rough_epochs=30, fine_epochs=30, decay="linear"):
        """
        Convenience method that runs a two-phase training: rough phase with large radius followed by
        a fine-tuning phase with small radius. Radius values are derived from the map size.
        :param data: dataset to use to train the model (np.array)
        :param rough_epochs: number of epochs for the rough phase (int)
        :param fine_epochs: number of epochs for the fine-tuning phase (int)
        :param decay: radius decay schedule. 'linear' or 'exponential'. (str)
        :return: the model trained (SOM)
        """
        data = _check_data(data)
        # Resolve auto mapsize early so we can derive radii
        data_norm = self.normalizer.normalize(data)
        if self.model_is_unfitted and self._mapsize == "auto":
            self._init_codebook(estimate_mapsize(data_norm), self._lattice, self._distance_metric)
        max_radius = max(self.codebook.n_rows, self.codebook.n_columns) / 2
        self.fit(data, rough_epochs, max_radius, max_radius / 4, decay=decay)
        self.fit(data, fine_epochs, max_radius / 4, 1, decay=decay)
        return self

    def predict(self, x):
        """
        Given a new dataset, if the model is trained, this function returns the
        bmu indices corresponding with the input data points.
        These indices correspond to the index over axis 0 of the self.codebook.matrix array
        :param x: input data set to use for inference (np.array)

        :returns: array of bmu indices (np.array)
        """
        if self.model_is_unfitted:
            raise ModelNotTrainedError
        x_norm = self.normalizer.transform(x)
        index, _ = find_bmu(codebook=self.codebook, input_matrix=x_norm, metric=self.metric)
        return index.astype(int)

    def calculate_quantization_error(self):
        """
        Calculates the quantization error of the model, i.e. the error which is originated by assigning the instances
        to the BMUs. It is calculated as the average Mean Absolute Error (MAE) between the instances and their assigned
        BMU.
        :return: the quantization error (float)
        """
        if self.model_is_unfitted:
            raise ModelNotTrainedError("The codebook of the model has not been initialized, you must call the fit "
                                       "method before calculating the quantization error.")
        neuron_values = self.codebook.matrix[find_bmu(self.codebook, self.data_norm,
                                                      metric=self.codebook.lattice.distance_metric)[0].astype(int)]
        quantization_error = np.mean(np.abs(neuron_values - self.data_norm))
        return quantization_error

    def calculate_vacancy_rate(self):
        """
        Calculates the vacancy rate of the model, i.e. the percentage of neurons that have no data points assigned.
        :return: the vacancy rate (float)
        """
        if self.model_is_unfitted:
            raise ModelNotTrainedError("The codebook of the model has not been initialized, you must call the fit "
                                       "method before calculating the vacancy rate.")
        bmu_indices = self.bmu[0].astype(int)
        n_active = len(np.unique(bmu_indices))
        return 1 - n_active / self.codebook.nnodes

    def calculate_topographic_error(self):
        """
        Calculates the topographic error of the model, i.e. the percentage of instances for which the BMU and the
        2nd BMU are not immediate neighbors.
        :return: the topographic error (float)
        """
        if self.model_is_unfitted:
            raise ModelNotTrainedError("The codebook of the model has not been initialized, you must call the fit "
                                       "method before calculating the topographic error.")
        metric = self.codebook.lattice.distance_metric
        bmus1 = find_bmu(self.codebook, self.data_norm, metric=metric, njb=self.n_jobs, nth=1)[0]
        bmus2 = find_bmu(self.codebook, self.data_norm, metric=metric, njb=self.n_jobs, nth=2)[0]
        neighs = [self.codebook.lattice.are_neighbor_indices(int(x1), int(x2)) for (x1, x2) in zip(bmus1, bmus2)]
        return 1 - np.mean(neighs)


def train_som(data, codebook, epochs, radiusin, radiusfin, neighborhood_f, distance_matrix, distance_metric, n_jobs,
              decay="linear", learning_rate=1.0, subsample_ratio=1.0):
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
    :param decay: radius decay schedule. 'linear' or 'exponential'. (str)
    :param learning_rate: blending factor for codebook update. 1.0 = full batch. (float)
    :param subsample_ratio: fraction of data per epoch. 1.0 = all data. (float)
    :return: bmus for each data instance (np.array
    """
    if decay == "exponential":
        radius = radiusin * (radiusfin / max(radiusin, 1e-10)) ** (np.arange(epochs) / max(epochs - 1, 1))
    else:
        radius = np.linspace(radiusin, radiusfin, epochs)

    n_samples = data.shape[0]
    use_subsample = subsample_ratio < 1.0
    subsample_size = max(1, int(n_samples * subsample_ratio))

    for i in range(epochs):
        if use_subsample:
            idx = np.random.choice(n_samples, subsample_size, replace=False)
            epoch_data = data[idx]
        else:
            epoch_data = data

        neighborhood = neighborhood_f.calculate(distance_matrix, radius[i], codebook.nnodes)
        bmu = find_bmu(codebook, epoch_data, metric=distance_metric, njb=n_jobs)
        new_codebook = update_codebook_voronoi(codebook, epoch_data, bmu, neighborhood, epoch_data.shape[0])
        if learning_rate < 1.0:
            codebook.matrix = learning_rate * new_codebook + (1 - learning_rate) * codebook.matrix
        else:
            codebook.matrix = new_codebook

    bmu = find_bmu(codebook, data, metric=distance_metric, njb=n_jobs)
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
    new_codebook = np.where(denominator != 0, np.divide(numerator, denominator), codebook.matrix)
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
    bmu[:, 0] = np.argpartition(d, nth - 1, axis=0)[nth - 1]  # BMU index
    bmu[:, 1] = np.partition(d, nth - 1, axis=0)[nth - 1]  # Distance
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
    chunks = list(batching([input_matrix], n=max(1, dlen // njb), return_incomplete_batches=True))

    with Pool(njb) as pool:
        bmus = pool.map(lambda chk: _chunk_based_bmu_find(instance=chk,
                                                          codebook=codebook.matrix,
                                                          distance_metric=metric,
                                                          nth=nth), chunks)
    bmus = np.asarray(list(itertools.chain(*bmus))).T
    return bmus

def _check_data(data):
    if np.isnan(data).any():
        raise InvalidValuesInDataSet("NaN values found in the data provided. Please "
                                     "remove them and rerun the function.")

    if np.isinf(data).any():
        raise InvalidValuesInDataSet("+/- Inf values found in the data provided. Please "
                                     "remove them and rerun the function.")

    return data
