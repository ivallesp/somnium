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


class TestToroidalRectLattice(TestCase):
    def test_dimension(self):
        lat = LatticeFactory.build("toroidal_rect")(n_rows=4, n_cols=5, distance_metric="euclidean")
        self.assertEqual(20, len(lat.coordinates))

    def test_wrapping_neighbors(self):
        # In a toroidal rect, edge nodes wrap to the opposite edge
        lat = LatticeFactory.build("toroidal_rect")(n_rows=4, n_cols=5, distance_metric="euclidean")
        # Node 0 (row=0, col=0) should be neighbor of node 4 (row=0, col=4) via column wrap
        self.assertTrue(lat.are_neighbor_indices(0, 4))
        # Node 0 (row=0, col=0) should be neighbor of node 15 (row=3, col=0) via row wrap
        self.assertTrue(lat.are_neighbor_indices(0, 15))

    def test_more_neighbors_than_flat(self):
        # Corner nodes in toroidal should have more neighbors than in flat
        flat = LatticeFactory.build("rect")(n_rows=4, n_cols=5, distance_metric="euclidean")
        toro = LatticeFactory.build("toroidal_rect")(n_rows=4, n_cols=5, distance_metric="euclidean")
        # Node 0 is a corner: 2 neighbors flat, 4 neighbors toroidal
        flat_n = sum(flat.are_neighbor_indices(0, j) for j in range(1, 20))
        toro_n = sum(toro.are_neighbor_indices(0, j) for j in range(1, 20))
        self.assertGreater(toro_n, flat_n)

    def test_all_nodes_same_neighbor_count(self):
        # In a toroidal rect lattice, every node should have exactly 4 neighbors
        lat = LatticeFactory.build("toroidal_rect")(n_rows=4, n_cols=5, distance_metric="euclidean")
        n = lat.n_rows * lat.n_cols
        for i in range(n):
            count = sum(lat.are_neighbor_indices(i, j) for j in range(n) if j != i)
            self.assertEqual(4, count, f"Node {i} has {count} neighbors, expected 4")


class TestToroidalHexaLattice(TestCase):
    def test_dimension(self):
        lat = LatticeFactory.build("toroidal_hexa")(n_rows=4, n_cols=5, distance_metric="euclidean")
        self.assertEqual(20, len(lat.coordinates))

    def test_more_neighbors_than_flat(self):
        flat = LatticeFactory.build("hexa")(n_rows=4, n_cols=5, distance_metric="euclidean")
        toro = LatticeFactory.build("toroidal_hexa")(n_rows=4, n_cols=5, distance_metric="euclidean")
        # Corner node should gain neighbors from wrapping
        flat_n = sum(flat.are_neighbor_indices(0, j) for j in range(1, 20))
        toro_n = sum(toro.are_neighbor_indices(0, j) for j in range(1, 20))
        self.assertGreaterEqual(toro_n, flat_n)

    def test_wrapped_distance_shorter(self):
        # Wrapped distance between opposite corners should be shorter than flat
        flat = LatticeFactory.build("hexa")(n_rows=6, n_cols=6, distance_metric="euclidean")
        toro = LatticeFactory.build("toroidal_hexa")(n_rows=6, n_cols=6, distance_metric="euclidean")
        flat_dist = flat.distances.reshape(36, 36)[0, 35]
        toro_dist = toro.distances.reshape(36, 36)[0, 35]
        self.assertLess(toro_dist, flat_dist)


class TestCylindricalRectLattice(TestCase):
    def test_dimension(self):
        lat = LatticeFactory.build("cylindrical_rect")(n_rows=4, n_cols=5, distance_metric="euclidean")
        self.assertEqual(20, len(lat.coordinates))

    def test_column_wrapping(self):
        # Columns wrap: node 0 (row=0,col=0) neighbors node 4 (row=0,col=4)
        lat = LatticeFactory.build("cylindrical_rect")(n_rows=4, n_cols=5, distance_metric="euclidean")
        self.assertTrue(lat.are_neighbor_indices(0, 4))

    def test_rows_do_not_wrap(self):
        # Rows do NOT wrap: node 0 (row=0,col=0) is NOT neighbor of node 15 (row=3,col=0)
        lat = LatticeFactory.build("cylindrical_rect")(n_rows=4, n_cols=5, distance_metric="euclidean")
        self.assertFalse(lat.are_neighbor_indices(0, 15))


class TestCylindricalHexaLattice(TestCase):
    def test_dimension(self):
        lat = LatticeFactory.build("cylindrical_hexa")(n_rows=4, n_cols=5, distance_metric="euclidean")
        self.assertEqual(20, len(lat.coordinates))

    def test_partial_wrapping(self):
        # Cylindrical should have more neighbors than flat but fewer than toroidal for corner nodes
        flat = LatticeFactory.build("hexa")(n_rows=4, n_cols=5, distance_metric="euclidean")
        cyl = LatticeFactory.build("cylindrical_hexa")(n_rows=4, n_cols=5, distance_metric="euclidean")
        toro = LatticeFactory.build("toroidal_hexa")(n_rows=4, n_cols=5, distance_metric="euclidean")
        flat_n = sum(flat.are_neighbor_indices(0, j) for j in range(1, 20))
        cyl_n = sum(cyl.are_neighbor_indices(0, j) for j in range(1, 20))
        toro_n = sum(toro.are_neighbor_indices(0, j) for j in range(1, 20))
        self.assertGreaterEqual(cyl_n, flat_n)
        self.assertGreaterEqual(toro_n, cyl_n)


class TestGetNeighborIndices(TestCase):
    def test_hexa_corner_has_3_neighbors(self):
        lattice = LatticeFactory.build("hexa")(n_rows=5, n_cols=6, distance_metric="euclidean")
        neighbors = lattice.get_neighbor_indices(0)
        self.assertEqual(len(neighbors), 3)

    def test_hexa_interior_has_6_neighbors(self):
        lattice = LatticeFactory.build("hexa")(n_rows=5, n_cols=6, distance_metric="euclidean")
        neighbors = lattice.get_neighbor_indices(13)  # interior node
        self.assertEqual(len(neighbors), 6)

    def test_rect_corner_has_2_neighbors(self):
        lattice = LatticeFactory.build("rect")(n_rows=5, n_cols=6, distance_metric="euclidean")
        neighbors = lattice.get_neighbor_indices(0)
        self.assertEqual(len(neighbors), 2)

    def test_neighbors_are_symmetric(self):
        lattice = LatticeFactory.build("hexa")(n_rows=4, n_cols=5, distance_metric="euclidean")
        for i in range(len(lattice.coordinates)):
            neighbors = lattice.get_neighbor_indices(i)
            for j in neighbors:
                self.assertIn(i, lattice.get_neighbor_indices(j))