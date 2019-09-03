import numpy as np

from sklearn.decomposition import PCA
from somnium.lattice import LatticeFactory


class Codebook(object):
    """
    Structure used to gather all the codebook-related information. It stores the following information:
    - the codebook matrix
    - the shapes of the maps
    - the lattice information
    - the initialization methods (random/PCA)
    :param mapsize: shape of the maps to be used in the SOM algorithm (tuple)
    :param lattice: hexagonal or rectangular lattice ("hexa" or "rect") (str)
    :param distance_metric: distance metric to be used in the lattice operations (str). The supported distance metrics
    are the same ones as the supported by scipy:
    - braycurtis
    - canberra
    - chebyshev
    - cityblock
    - correlation
    - cosine
    - euclidean
    - jensenshannon
    - mahalanobis
    - minkowski
    - seuclidean
    - sqeuclidean
    - wminkowski
    More info at https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    """
    def __init__(self, mapsize, lattice='hexa', distance_metric="sqeuclidean"):
        self.mapsize = [1, np.max(mapsize)] if 1 == np.min(mapsize) else mapsize
        self.nnodes = mapsize[0]*mapsize[1]
        self.matrix = np.asarray(self.mapsize)
        self.initialized = False
        self.n_rows, self.n_columns = self.mapsize
        self.lattice = LatticeFactory.build(lattice)(n_rows=self.n_rows,
                                                     n_cols=self.n_columns,
                                                     distance_metric=distance_metric)

    def random_initialization(self, data):
        """
        Initializes the codebook using a random gaussian distribution.
        :param data: data to use for the initialization
        :returns: initialized matrix with same dimension as input data
        """
        mn = np.tile(np.min(data, axis=0), (self.nnodes, 1))
        mx = np.tile(np.max(data, axis=0), (self.nnodes, 1))
        self.matrix = mn + (mx-mn)*(np.random.rand(self.nnodes, data.shape[1]))
        self.initialized = True

    def pca_linear_initialization(self, data):
        """
        Initializes the codebook just by using the first two first eigen vals and
        eigenvectors. Further, it creates a linear combination of them in the new map by
        giving values from -1 to 1 in each

        X = UsigmaWT
        XTX = Wsigma^2WT
        T = XW = Usigma

        // Transformed by W EigenVector, can be calculated by multiplication
        // PC matrix by eigenval too
        // Further, we can get lower ranks by using just few of the eigen
        // vevtors

        T(2) = U(2)sigma(2) = XW(2) ---> 2 is the number of selected
        eigenvectors

        (*) Note that 'X' is the covariance matrix of original data

        :param data: data to use for the initialization
        :returns: initialized matrix with same dimension as input data
        """
        cols = self.mapsize[1]
        coord = None
        pca_components = None

        if np.min(self.mapsize) > 1:
            coord = np.zeros((self.nnodes, 2))
            pca_components = 2

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i / cols)  # x
                coord[i, 1] = int(i % cols)  # y

        elif np.min(self.mapsize) == 1:
            coord = np.zeros((self.nnodes, 1))
            pca_components = 1

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i % cols)  # y

        mx = np.max(coord, axis=0)
        mn = np.min(coord, axis=0)
        coord = (coord - mn)/(mx-mn)
        coord = (coord - .5)*2
        me = np.mean(data, 0)
        data = (data - me)
        tmp_matrix = np.tile(me, (self.nnodes, 1))

        # Randomized PCA is scalable
        pca = PCA(n_components=pca_components, svd_solver='randomized')

        pca.fit(data)
        eigvec = pca.components_
        eigval = pca.explained_variance_
        norms = np.sqrt(np.einsum('ij,ij->i', eigvec, eigvec))
        eigvec = ((eigvec.T/norms)*eigval).T

        for j in range(self.nnodes):
            for i in range(eigvec.shape[0]):
                tmp_matrix[j, :] = tmp_matrix[j, :] + coord[j, i]*eigvec[i, :]

        self.matrix = np.around(tmp_matrix, decimals=6)
        self.initialized = True
