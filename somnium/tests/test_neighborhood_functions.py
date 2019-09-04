from unittest import TestCase
import numpy as np

from somnium.neighborhood import NeighborhoodFactory
from somnium.tests.util import non_increasing, strictly_decreasing


class TestNeighborhood(TestCase):
    def test_gaussian_neighborhood(self):
        # Check monotonicity
        x = np.array(list(range(9)))
        neigh_f = NeighborhoodFactory.build("gaussian")
        neigh = neigh_f.calculate(distance_matrix=x, radius=1, dim=3).reshape(-1)
        self.assertTrue(strictly_decreasing(neigh))

    def test_bubble_neighborhood(self):
        # Check monotonicity
        x = np.array(list(range(9)))
        neigh_f = NeighborhoodFactory.build("bubble")
        neigh = neigh_f.calculate(distance_matrix=x, radius=3, dim=3).reshape(-1)
        self.assertTrue(non_increasing(neigh))
        self.assertAlmostEqual(1, neigh[x <= 3].mean())
        self.assertAlmostEqual(0, neigh[x > 3].mean())

    def test_cut_gaussian_neighborhood(self):
        # Check monotonicity
        x = np.array(list(range(9)))
        neigh_f = NeighborhoodFactory.build("cut_gaussian")
        neigh = neigh_f.calculate(distance_matrix=x, radius=3, dim=3).reshape(-1)
        self.assertTrue(non_increasing(neigh))

        # Check it is exactly a cut-gaussian, compare with gaussian
        neigh_f_gaussian = NeighborhoodFactory.build("gaussian")
        neigh_gaussian = neigh_f_gaussian.calculate(distance_matrix=x, radius=3, dim=3).reshape(-1)
        self.assertAlmostEqual(0, (neigh_gaussian-neigh)[x <= 3].sum())
        self.assertNotEqual(0, (neigh_gaussian - neigh)[x > 3].sum())


    def test_epanechicov_neighborhood(self):
        # Check monotonicity
        x = np.array(list(range(9)))
        neigh_f = NeighborhoodFactory.build("epanechicov")
        neigh = neigh_f.calculate(distance_matrix=x, radius=5, dim=3).reshape(-1)
        self.assertTrue(non_increasing(neigh))

        # Assure it cuts in the correct place
        self.assertTrue((neigh[x <= 5]>0).all())
        self.assertAlmostEqual(0, neigh[x >= 5].mean())
        self.assertLessEqual(0, neigh.min())