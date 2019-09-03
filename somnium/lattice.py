import inspect
import sys
import scipy as sp
import numpy as np


class LatticeFactory(object):
    """
    Factory class for lattices
    """
    @staticmethod
    def build(lattice):
        """
        Looks for the existing lattices and returns the chosen one if exists. If it doesn't, it throws an exception
        :param name: name of the lattice
        :return: lattice class (class)
        """
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and lattice == obj.name:
                    return obj
        else:
            raise Exception(
                "Unsupported latice shape '%s'" % lattice)


class Lattice:
    def __init__(self, *args, **kwargs):
        """
        Base class for lattices.
        """
        self.coordinates = self.generate_lattice(n_rows=self.n_rows, n_cols=self.n_cols)
        self.distances = self.compute_distance_matrix(distance_metric=self.distance_metric)

    @staticmethod
    def generate_lattice(rows, cols):
        """
        Function in charge of generating the lattice coordinates.
        :param rows: number of rows in the lattice (int)
        :param cols: number of columns in the lattice (int)
        :return: list of coordinates (list)
        """
        raise NotImplementedError

    def are_neighbors(self, u1, u2):
        """
        Determines if two units are neighbors. In that case it returns True, otherwise it returns False
        :param u1: coordinates of the first unit (int)
        :param u2: coordinates of the second unit (init)
        :return: boolean value indicating if the units are neighbors (bool)
        """
        raise NotImplementedError

    def are_neighbor_indices(self, i1, i2):
        """
        Determines if two units are neighbors. In that case it returns True, otherwise it returns False
        :param i1: index of the first unit (int)
        :param i2: index of the second unit (init)
        :return: boolean value indicating if the units are neighbors (bool)
        """
        u1 = self.coordinates[i1]
        u2 = self.coordinates[i2]
        return self.are_neighbors(u1, u2)

    def compute_distance_matrix(self, distance_metric):
        """
        Given a distance metric, it computes the distance matrix between the lattice planar coordinates.
        :param distance_metric: name of the distance metric to use to compute the distances (str)
        :return: distance matrix (np.array)
        """
        dist = sp.spatial.distance.pdist(self.coordinates, metric=distance_metric)
        dist = sp.spatial.distance.squareform(dist)
        dist = dist.reshape(self.n_rows * self.n_cols, self.n_rows, self.n_cols)
        return dist

class HexaLattice(Lattice):
    name = "hexa"

    def __init__(self, n_rows, n_cols, distance_metric):
        """
        Hexagonal lattice
        :param n_rows: number of rows (int)
        :param n_cols: number of columns (int)
        :param distance_metric: name of the distance metric to use to compute the distances (str)
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.distance_metric = distance_metric
        super().__init__()

    @staticmethod
    def generate_lattice(n_rows, n_cols):
        """
        Function in charge of generating the lattice coordinates.
        :param rows: number of rows in the lattice (int)
        :param cols: number of columns in the lattice (int)
        :return: list of coordinates (list)
        """
        x_coord = []
        y_coord = []
        for i in range(n_rows):
            for j in range(n_cols):
                x_coord.append(i * 1.5)
                y_coord.append(np.sqrt(2 / 3) * (2 * j + (1 + i) % 2))
        coordinates = np.column_stack([x_coord, y_coord])
        coordinates[:, 1] = coordinates[:, 1] * np.sqrt(3) / np.sqrt(8) + 0.5
        coordinates[:, 0] = coordinates[:, 0] * np.sqrt(3) / 3
        return coordinates



    def are_neighbors(self, u1, u2):
        """
        Determines if two units are neighbors. In that case it returns True, otherwise it returns False
        :param u1: coordinates of the first unit (int)
        :param u2: coordinates of the second unit (init)
        :return: boolean value indicating if the units are neighbors (bool)
        """
        l2 = np.sqrt((u1[0] - u2[0]) ** 2 + (u1[1] - u2[1]) ** 2)  # Euclidean distance
        return l2 < 2


class RectLattice(Lattice):
    name = "rect"

    def __init__(self, n_rows, n_cols, distance_metric):
        """
        Rectangular lattice
        :param n_rows: number of rows (int)
        :param n_cols: number of columns (int)
        :param distance_metric: name of the distance metric to use to compute the distances (str)
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.distance_metric = distance_metric
        super().__init__()

    @staticmethod
    def generate_lattice(n_rows, n_cols):
        """
        Function in charge of generating the lattice coordinates.
        :param rows: number of rows in the lattice (int)
        :param cols: number of columns in the lattice (int)
        :return: list of coordinates (list)
        """
        x_coord = []
        y_coord = []
        for i in range(n_rows):
            for j in range(n_cols):
                x_coord.append(i)
                y_coord.append(j)
        coordinates = np.column_stack([x_coord, y_coord])
        return coordinates

    def are_neighbors(self, u1, u2):
        """
        Determines if two units are neighbors. In that case it returns True, otherwise it returns False
        :param u1: coordinates of the first unit (int)
        :param u2: coordinates of the second unit (init)
        :return: boolean value indicating if the units are neighbors (bool)
        """
        l2 = np.sqrt((u1[0] - u2[0]) ** 2 + (u1[1] - u2[1]) ** 2)  # Euclidean distance
        return l2 <= 1