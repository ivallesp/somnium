import numpy as np
import inspect
import sys
import scipy as sp
import numpy as np


class LatticeFactory(object):
    @staticmethod
    def build(lattice):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and lattice == obj.name:
                    return obj
        else:
            raise Exception(
                "Unsupported latice shape '%s'" % lattice)


class Lattice:
    def __init__(self, *args, **kwargs):
        self.coordinates = self.generate_lattice()
        self.distances = self.compute_distance_matrix(distance_metric=self.distance_metric)

    def generate_lattice(self):
        raise NotImplementedError

    def are_neighbors(self, u1, u2):
        raise NotImplementedError

    def are_neighbor_indices(self, i1, i2):
        u1 = self.coordinates[i1]
        u2 = self.coordinates[i2]
        return self.are_neighbors(u1, u2)

    def compute_distance_matrix(self, distance_metric):
        dist = sp.spatial.distance.pdist(self.coordinates, metric=distance_metric)
        dist = sp.spatial.distance.squareform(dist)
        dist = dist.reshape(self.n_rows * self.n_cols, self.n_rows, self.n_cols)
        return dist


class HexaLattice(Lattice):
    name = "hexa"

    def __init__(self, n_rows, n_cols, distance_metric):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.distance_metric = distance_metric
        super().__init__()

    def generate_lattice(self):
        x_coord = []
        y_coord = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                x_coord.append(i * 1.5)
                y_coord.append(np.sqrt(2 / 3) * (2 * j + (1 + i) % 2))
        coordinates = np.column_stack([x_coord, y_coord])
        return coordinates

    def are_neighbors(self, u1, u2):
        l2 = np.sqrt((u1[0] - u2[0]) ** 2 + (u1[1] - u2[1]) ** 2)  # Euclidean distance
        return l2 < 2


class RectLattice(Lattice):
    name = "rect"

    def __init__(self, n_rows, n_cols, distance_metric):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.distance_metric = distance_metric
        super().__init__()

    def generate_lattice(self):
        x_coord = []
        y_coord = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                x_coord.append(i)
                y_coord.append(j)
        coordinates = np.column_stack([x_coord, y_coord])
        return coordinates

    def are_neighbors(self, u1, u2):
        l2 = np.sqrt((u1[0] - u2[0]) ** 2 + (u1[1] - u2[1]) ** 2)  # Euclidean distance
        return l2 <= 1