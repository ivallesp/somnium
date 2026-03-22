import inspect
import sys
from scipy.spatial.distance import pdist, squareform
import numpy as np
from somnium.exceptions import LatticeTypeNotFound

epsilon = 1e-6


class LatticeFactory(object):
    """
    Factory class for lattices
    """
    @staticmethod
    def build(lattice):
        """
        Looks for the existing lattices and returns the chosen one if exists. If it doesn't, it throws an exception
        :param lattice: name of the lattice
        :return: lattice class (class)
        """
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and lattice == obj.name:
                    return obj
        else:
            raise LatticeTypeNotFound(
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

    def get_neighbor_indices(self, idx):
        """
        Returns the indices of all lattice neighbors of the given unit.
        :param idx: index of the unit (int)
        :return: list of neighbor indices (list of int)
        """
        n_units = len(self.coordinates)
        return [j for j in range(n_units) if j != idx and self.are_neighbor_indices(idx, j)]

    def compute_distance_matrix(self, distance_metric):
        """
        Given a distance metric, it computes the distance matrix between the lattice planar coordinates.
        :param distance_metric: name of the distance metric to use to compute the distances (str)
        :return: distance matrix (np.array)
        """
        dist = pdist(self.coordinates, metric=distance_metric)
        dist = squareform(dist)
        dist = dist.reshape(self.n_rows * self.n_cols, self.n_rows, self.n_cols)
        return dist

    def _compute_extents(self):
        """Compute the periodic extent of the lattice in each dimension."""
        coords = self.coordinates
        unique_x = np.unique(np.round(coords[:, 0], 10))
        unique_y = np.unique(np.round(coords[:, 1], 10))
        step_x = np.min(np.diff(unique_x)) if len(unique_x) > 1 else 1.0
        step_y = np.min(np.diff(unique_y)) if len(unique_y) > 1 else 1.0
        self._extent_x = np.max(coords[:, 0]) - np.min(coords[:, 0]) + step_x
        self._extent_y = np.max(coords[:, 1]) - np.min(coords[:, 1]) + step_y

    def _compute_wrapped_distance_matrix(self, wrap_rows, wrap_cols):
        """Compute pairwise distance matrix with wrapping in specified dimensions."""
        self._compute_extents()
        coords = self.coordinates
        n = len(coords)
        dist = np.zeros((n, n))
        for i in range(n):
            dx = np.abs(coords[i, 0] - coords[:, 0])
            dy = np.abs(coords[i, 1] - coords[:, 1])
            if wrap_rows:
                dx = np.minimum(dx, self._extent_x - dx)
            if wrap_cols:
                dy = np.minimum(dy, self._extent_y - dy)
            dist[i] = np.sqrt(dx ** 2 + dy ** 2)
        return dist.reshape(self.n_rows * self.n_cols, self.n_rows, self.n_cols)

    def _wrapped_distance(self, u1, u2, wrap_rows, wrap_cols):
        """Euclidean distance between two points with optional wrapping."""
        dx = abs(u1[0] - u2[0])
        dy = abs(u1[1] - u2[1])
        if wrap_rows:
            dx = min(dx, self._extent_x - dx)
        if wrap_cols:
            dy = min(dy, self._extent_y - dy)
        return np.sqrt(dx ** 2 + dy ** 2)


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
        :param n_rows: number of rows in the lattice (int)
        :param n_cols: number of columns in the lattice (int)
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
        l2 = np.sqrt((u1[0] - u2[0]) ** 2 + (u1[1] - u2[1]) ** 2)
        return l2 <= (1 + epsilon)


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
        :param n_rows: number of rows in the lattice (int)
        :param n_cols: number of columns in the lattice (int)
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
        return l2 <= (1 + epsilon)


class ToroidalHexaLattice(HexaLattice):
    name = "toroidal_hexa"

    def compute_distance_matrix(self, distance_metric):
        return self._compute_wrapped_distance_matrix(wrap_rows=True, wrap_cols=True)

    def are_neighbors(self, u1, u2):
        return self._wrapped_distance(u1, u2, wrap_rows=True, wrap_cols=True) <= (1 + epsilon)


class ToroidalRectLattice(RectLattice):
    name = "toroidal_rect"

    def compute_distance_matrix(self, distance_metric):
        return self._compute_wrapped_distance_matrix(wrap_rows=True, wrap_cols=True)

    def are_neighbors(self, u1, u2):
        return self._wrapped_distance(u1, u2, wrap_rows=True, wrap_cols=True) <= (1 + epsilon)


class CylindricalHexaLattice(HexaLattice):
    name = "cylindrical_hexa"

    def compute_distance_matrix(self, distance_metric):
        return self._compute_wrapped_distance_matrix(wrap_rows=False, wrap_cols=True)

    def are_neighbors(self, u1, u2):
        return self._wrapped_distance(u1, u2, wrap_rows=False, wrap_cols=True) <= (1 + epsilon)


class CylindricalRectLattice(RectLattice):
    name = "cylindrical_rect"

    def compute_distance_matrix(self, distance_metric):
        return self._compute_wrapped_distance_matrix(wrap_rows=False, wrap_cols=True)

    def are_neighbors(self, u1, u2):
        return self._wrapped_distance(u1, u2, wrap_rows=False, wrap_cols=True) <= (1 + epsilon)
