import itertools
import pickle

import numpy as np

from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA

from somnium.codebook import Codebook
from somnium.normalization import NormalizerFactory
from somnium.neighborhood import NeighborhoodFactory
from somnium.util import batching
from somnium.exceptions import ModelNotTrainedError, InvalidValuesInDataSet


def estimate_mapsize(data, scale=5, max_aspect_ratio=None):
    """
    Estimates a good map size for the given data using a heuristic: total neurons ≈ scale * sqrt(N),
    aspect ratio derived from the ratio of the first two PCA eigenvalues.
    :param data: input data matrix (np.array)
    :param scale: multiplier for sqrt(N) to control total neurons (float, default 5)
    :param max_aspect_ratio: if set, caps the aspect ratio to prevent very elongated maps.
    None (default) = no cap. (float or None)
    :return: (n_rows, n_columns) tuple
    """
    n_samples = data.shape[0]
    total_neurons = int(scale * np.sqrt(n_samples))
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
                 distance_metric="euclidean", n_jobs=1, initialization="random",
                 mapsize_scale=5):
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
        self._mapsize_scale = mapsize_scale
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

    def save(self, path):
        """
        Saves the model to disk using pickle.
        :param path: file path to save to (str or Path)
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Loads a model from disk.
        :param path: file path to load from (str or Path)
        :return: the loaded SOM model
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Expected a SOM instance, got {type(model).__name__}")
        return model

    def fit(self, data, epochs, radiusin, radiusfin, decay="linear",
            learning_rate=1.0, subsample_ratio=1.0, collect_history=False):
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
        :param collect_history: if True, record per-epoch quantization_error, topographic_error,
        and vacancy_rate in self.history_ (dict of lists). Appends to existing history if present.
        Note: this adds overhead since TE requires two full BMU searches per epoch. (bool)
        :return: the model trained (SOM)
        """
        data = _check_data(data)
        data_norm = self.normalizer.normalize(data)
        self.data_norm = data_norm
        _dlen = data_norm.shape[0]
        if self.model_is_unfitted:
            if self._mapsize == "auto":
                self._init_codebook(estimate_mapsize(data_norm, scale=self._mapsize_scale), self._lattice, self._distance_metric)
            if self.initialization == "pca":
                self.codebook.pca_linear_initialization(data_norm)
            else:
                self.codebook.random_initialization(data_norm)
            self.model_is_unfitted = False
        if collect_history and not hasattr(self, 'history_'):
            self.history_ = {"quantization_error": [], "topographic_error": [], "vacancy_rate": []}
        self.bmu = train_som(data_norm, self.codebook, epochs, radiusin, radiusfin,
                             neighborhood_f=self.neighborhood_calculator,
                             distance_matrix=self.distance_matrix,
                             distance_metric=self.codebook.lattice.distance_metric,
                             n_jobs=self.n_jobs, decay=decay,
                             learning_rate=learning_rate,
                             subsample_ratio=subsample_ratio,
                             collect_history=self.history_ if collect_history else None,
                             lattice=self.codebook.lattice)
        # Update the bmus at the end
        self.bmu = find_bmu(self.codebook,
                            data_norm,
                            metric=self.codebook.lattice.distance_metric,
                            njb=self.n_jobs)

        return self

    def fit_auto(self, data, rough_epochs=400, fine_epochs=400):
        """
        Convenience method that runs an optimized two-phase training schedule.

        Rough phase uses linear radius decay for uniform topology building.
        Fine phase uses exponential decay for precise convergence.
        Radii are derived from the map size: rough covers ~1/3 of the map
        diameter down to radius 3, fine goes from 3 down to 1.

        :param data: dataset to use to train the model (np.array)
        :param rough_epochs: number of epochs for the rough phase (int, default 400)
        :param fine_epochs: number of epochs for the fine-tuning phase (int, default 400)
        :return: the model trained (SOM)
        """
        data = _check_data(data)
        # Resolve auto mapsize early so we can derive radii
        data_norm = self.normalizer.normalize(data)
        if self.model_is_unfitted and self._mapsize == "auto":
            self._init_codebook(estimate_mapsize(data_norm, scale=self._mapsize_scale), self._lattice, self._distance_metric)
        max_dim = max(self.codebook.n_rows, self.codebook.n_columns)
        rough_start = max(max_dim / 3, 4)
        transition = 3
        self.fit(data, rough_epochs, rough_start, transition,
                 decay="linear")
        self.fit(data, fine_epochs, transition, 1,
                 decay="exponential")
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
        bmu1, bmu2 = find_bmu_top2(self.codebook, self.data_norm, metric=metric, njb=self.n_jobs)
        neighbor_matrix = self.codebook.lattice.neighbor_matrix
        neighs = neighbor_matrix[bmu1[0].astype(int), bmu2[0].astype(int)]
        return float(1 - np.mean(neighs))


def train_som(data, codebook, epochs, radiusin, radiusfin, neighborhood_f, distance_matrix, distance_metric, n_jobs,
              decay="linear", learning_rate=1.0, subsample_ratio=1.0, collect_history=None, lattice=None):
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
    :param collect_history: if not None, a dict with keys 'quantization_error', 'topographic_error',
    'vacancy_rate' to which per-epoch values are appended. (dict or None)
    :param lattice: lattice object, required when collect_history is not None (for TE computation).
    :return: bmus for each data instance (np.array)
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

        if collect_history is not None:
            full_bmu1, full_bmu2 = find_bmu_top2(codebook, data, metric=distance_metric, njb=n_jobs)
            # QE: mean distance from each data point to its BMU
            neuron_values = codebook.matrix[full_bmu1[0].astype(int)]
            qe = float(np.mean(np.abs(neuron_values - data)))
            # VR: fraction of neurons with no hits
            n_active = len(np.unique(full_bmu1[0].astype(int)))
            vr = float(1 - n_active / codebook.nnodes)
            # TE: fraction of data where 1st and 2nd BMU are not neighbors
            neighbor_matrix = lattice.neighbor_matrix
            neighs = neighbor_matrix[full_bmu1[0].astype(int), full_bmu2[0].astype(int)]
            te = float(1 - np.mean(neighs))
            collect_history["quantization_error"].append(qe)
            collect_history["topographic_error"].append(te)
            collect_history["vacancy_rate"].append(vr)

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


def _chunk_based_bmu_find_top2(instance, codebook, distance_metric):
    """
    Finds the 1st and 2nd BMU for each instance in a single cdist call.
    :param instance: batch of instances (list containing one np.array)
    :param codebook: codebook matrix (np.array)
    :param distance_metric: distance metric (str)
    :return: array of shape (dlen, 4): [bmu1_idx, bmu1_dist, bmu2_idx, bmu2_dist]
    """
    assert len(instance) == 1
    instance = instance[0]
    dlen = instance.shape[0]
    d = cdist(codebook, instance, metric=distance_metric)
    top2_idx = np.argpartition(d, 1, axis=0)[:2]
    top2_dist = np.partition(d, 1, axis=0)[:2]
    result = np.empty((dlen, 4))
    result[:, 0] = top2_idx[0]
    result[:, 1] = top2_dist[0]
    result[:, 2] = top2_idx[1]
    result[:, 3] = top2_dist[1]
    return result


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


def find_bmu_top2(codebook, input_matrix, metric, njb=1):
    """
    Finds both the 1st and 2nd best matching units in a single pass (one cdist per chunk).
    :param codebook: codebook object (somnium.codebook.Codebook)
    :param input_matrix: input data matrix (np.array)
    :param metric: distance metric name (str)
    :param njb: number of parallel jobs (int)
    :returns: tuple (bmu1, bmu2) where each is a (2, dlen) array of [indices, distances]
    """
    dlen = input_matrix.shape[0]
    njb = cpu_count() if njb == -1 else njb
    chunks = list(batching([input_matrix], n=max(1, dlen // njb), return_incomplete_batches=True))

    with Pool(njb) as pool:
        results = pool.map(lambda chk: _chunk_based_bmu_find_top2(
            instance=chk, codebook=codebook.matrix, distance_metric=metric), chunks)
    combined = np.vstack(list(itertools.chain(results)))
    bmu1 = np.array([combined[:, 0], combined[:, 1]])
    bmu2 = np.array([combined[:, 2], combined[:, 3]])
    return bmu1, bmu2


def _check_data(data):
    if np.isnan(data).any():
        raise InvalidValuesInDataSet("NaN values found in the data provided. Please "
                                     "remove them and rerun the function.")

    if np.isinf(data).any():
        raise InvalidValuesInDataSet("+/- Inf values found in the data provided. Please "
                                     "remove them and rerun the function.")

    return data
