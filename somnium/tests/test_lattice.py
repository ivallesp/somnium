from unittest import TestCase
import numpy as np
import math

from somnium.lattice import LatticeFactory
from scipy.spatial.distance import pdist, squareform
from itertools import combinations, product, compress
from somnium.tests.util import euclidean_distance


class TestRectLattice(TestCase):
    def test_dimension(self):
        lat = LatticeFactory.build("rect")(n_rows=2, n_cols=3, distance_metric="euclidean")
        self.assertEqual(6, len(lat.coordinates))
        self.assertEqual(2, lat.n_rows)
        self.assertEqual(3, lat.n_cols)

    def test_distances(self):
        lat = LatticeFactory.build("rect")(n_rows=2, n_cols=3, distance_metric="euclidean")
        pairs = list(product(lat.coordinates, lat.coordinates))
        dist = np.array([euclidean_distance(x=u1, y=u2) for (u1, u2) in pairs])
        dist = dist.reshape(6,2,3)
        self.assertTrue(np.allclose(dist, lat.distances))

    def test_ordering(self):
        lat = LatticeFactory.build("rect")(n_rows=2, n_cols=3, distance_metric="euclidean")
        self.assertTrue(lat.distances[0, 0, 0] == 0)
        self.assertTrue(lat.distances[1, 0, 1] == 0)
        self.assertTrue(lat.distances[2, 0, 2] == 0)
        self.assertTrue(lat.distances[5, 1, 2] == 0)

    def test_n_neighbors(self):
        lat = LatticeFactory.build("rect")(n_rows=4, n_cols=3, distance_metric="euclidean")
        dist_matrix = squareform(pdist(lat.coordinates))
        n_neighbors = set(np.sum(np.isclose(dist_matrix, 1), axis=0))
        self.assertEqual({2,3,4}, n_neighbors)

    def test_neighborhood_method(self):
        lat = LatticeFactory.build("rect")(n_rows=4, n_cols=7, distance_metric="euclidean")
        pairs = list(combinations(lat.coordinates, 2))
        neighbors = [euclidean_distance(x=u1, y=u2)==1 for (u1, u2) in pairs]
        neighbor_pairs = list(compress(pairs, neighbors))
        not_neighbor_pairs = list(compress(pairs, [not(n) for n in neighbors]))
        self.assertTrue(all([lat.are_neighbors(*x) for x in neighbor_pairs]))
        self.assertTrue(not(any([lat.are_neighbors(*x) for x in not_neighbor_pairs])))

    def test_neighborhood_method_cherrypick(self):
        lat = LatticeFactory.build("rect")(n_rows=7, n_cols=8, distance_metric="euclidean")
        center = 14
        neighbors = [6, 13, 15, 22]
        self.assertTrue(all([lat.are_neighbor_indices(center, n) for n in neighbors]))
        lat = LatticeFactory.build("rect")(n_rows=6, n_cols=7, distance_metric="euclidean")
        center = 8
        neighbors = [1, 7, 9, 15]
        self.assertTrue(all([lat.are_neighbor_indices(center, n) for n in neighbors]))
        center = 15
        neighbors = [8 ,14, 16, 22]
        self.assertTrue(all([lat.are_neighbor_indices(center, n) for n in neighbors]))


class TestHexaLattice(TestCase):
    def test_dimension(self):
        lat = LatticeFactory.build("hexa")(n_rows=2, n_cols=3, distance_metric="euclidean")
        self.assertEqual(6, len(lat.coordinates))
        self.assertEqual(2, lat.n_rows)
        self.assertEqual(3, lat.n_cols)

    def test_distances(self):
        lat = LatticeFactory.build("hexa")(n_rows=2, n_cols=3, distance_metric="euclidean")
        pairs = list(product(lat.coordinates, lat.coordinates))
        dist = np.array([euclidean_distance(x=u1, y=u2) for (u1, u2) in pairs])
        dist = dist.reshape(6, 2, 3)
        self.assertTrue(np.allclose(dist, lat.distances))

    def test_ordering(self):
        lat = LatticeFactory.build("hexa")(n_rows=2, n_cols=3, distance_metric="euclidean")
        self.assertTrue(lat.distances[0, 0, 0] == 0)
        self.assertTrue(lat.distances[1, 0, 1] == 0)
        self.assertTrue(lat.distances[2, 0, 2] == 0)
        self.assertTrue(lat.distances[5, 1, 2] == 0)

    def test_n_neighbors(self):
        lat = LatticeFactory.build("hexa")(n_rows=4, n_cols=3, distance_metric="euclidean")
        dist_matrix = squareform(pdist(lat.coordinates))
        n_neighbors = set(np.sum(np.isclose(dist_matrix, 1), axis=0))
        self.assertEqual({2,3,4,5,6}, n_neighbors)

    def test_neighborhood_method_in_batch(self):
        lat = LatticeFactory.build("hexa")(n_rows=4, n_cols=7, distance_metric="euclidean")
        pairs = list(combinations(lat.coordinates, 2))
        neighbors = [math.isclose(a=euclidean_distance(x=u1, y=u2), b=1) for (u1, u2) in pairs]
        neighbor_pairs = list(compress(pairs, neighbors))
        not_neighbor_pairs = list(compress(pairs, [not(n) for n in neighbors]))
        self.assertTrue(all([lat.are_neighbors(*x) for x in neighbor_pairs]))
        self.assertTrue(not(any([lat.are_neighbors(*x) for x in not_neighbor_pairs])))

    def test_neighborhood_method_cherrypick(self):
        lat = LatticeFactory.build("hexa")(n_rows=7, n_cols=8, distance_metric="euclidean")
        center = 14
        neighbors = [5, 6, 13, 15, 21, 22]
        self.assertTrue(all([lat.are_neighbor_indices(center, n) for n in neighbors]))
        lat = LatticeFactory.build("hexa")(n_rows=6, n_cols=7, distance_metric="euclidean")
        center = 8
        neighbors = [0, 1, 7, 9, 14, 15]
        self.assertTrue(all([lat.are_neighbor_indices(center, n) for n in neighbors]))
        center = 15
        neighbors = [8, 9, 14, 16, 22, 23]
        self.assertTrue(all([lat.are_neighbor_indices(center, n) for n in neighbors]))