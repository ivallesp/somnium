from unittest import TestCase
import numpy as np

from somnium.codebook import Codebook


class TestCodebook(TestCase):
    def test_dimensions(self):
        codebook = Codebook(mapsize=[7, 3], lattice="hexa", distance_metric="euclidean")
        self.assertEqual([7, 3], codebook.mapsize)
        self.assertEqual(21, codebook.nnodes)
        self.assertIsNone(codebook.matrix)  # Not initialized
        self.assertFalse(codebook.initialized)

    def test_random_initialization(self):
        codebook = Codebook(mapsize=[6, 7], lattice="rect", distance_metric="euclidean")
        codebook.random_initialization(data=np.random.rand(100, 3))
        self.assertTrue(codebook.initialized)
        self.assertEqual((42, 3), codebook.matrix.shape)
        self.assertLessEqual(0, codebook.matrix.min())
        self.assertGreaterEqual(1, codebook.matrix.max())

    def test_pca_initialization(self):
        codebook = Codebook(mapsize=[6, 7], lattice="rect", distance_metric="euclidean")
        codebook.pca_linear_initialization(data=np.random.rand(100, 3))
        self.assertTrue(codebook.initialized)
        self.assertEqual((42, 3), codebook.matrix.shape)
        self.assertLessEqual(0, codebook.matrix.min())
        self.assertGreaterEqual(1, codebook.matrix.max())

    def test_pca_initialization_unitary_shape(self):
        codebook = Codebook(mapsize=[1, 7], lattice="rect", distance_metric="euclidean")
        codebook.pca_linear_initialization(data=np.random.rand(100, 3))
        self.assertTrue(codebook.initialized)
        self.assertEqual((7, 3), codebook.matrix.shape)
        self.assertLessEqual(0, codebook.matrix.min())
        self.assertGreaterEqual(1, codebook.matrix.max())
