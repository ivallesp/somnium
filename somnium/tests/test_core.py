from unittest import TestCase
import numpy as np
import random

from somnium.core import SOM, find_bmu


class TestGeneralTraining(TestCase):
    def test_fit_rect(self):
        # Check that the training process effectively reduces the errors
        data = np.random.rand(100, 10)
        model = SOM(neighborhood="gaussian", normalization="standard", mapsize=[15, 10], lattice="rect",
                    distance_metric="euclidean", n_jobs=1)
        model.codebook.random_initialization(data)  # Manually initialize the codebook
        model.data_norm = model.normalizer.normalize(data)
        e_q, e_t = model.calculate_quantization_error(), model.calculate_topographic_error()
        f1_0 = 1/(1/e_q + 1/e_t)

        model.fit(data, epochs=10, radiusin=10, radiusfin=3)
        e_q, e_t = model.calculate_quantization_error(), model.calculate_topographic_error()
        f1_1 = 1/(1/e_q + 1/e_t)

        self.assertGreater(f1_0, f1_1)

    def test_fit_hexa(self):
        # Check that the training process effectively reduces the errors
        data = np.random.rand(100, 10)
        model = SOM(neighborhood="gaussian", normalization="standard", mapsize=[15, 10], lattice="hexa",
                    distance_metric="euclidean", n_jobs=1)
        model.codebook.random_initialization(data)  # Manually initialize the codebook
        model.data_norm = model.normalizer.normalize(data)
        e_q, e_t = model.calculate_quantization_error(), model.calculate_topographic_error()
        f1_0 = 1/(1/e_q + 1/e_t)

        model.fit(data, epochs=10, radiusin=10, radiusfin=3)
        e_q, e_t = model.calculate_quantization_error(), model.calculate_topographic_error()
        f1_1 = 1/(1/e_q + 1/e_t)

        self.assertGreater(f1_0, f1_1)

    def test_fit_different_parameters(self):
        # Check that the training process effectively reduces the errors for multiple metrics.
        # The criterion has been relaxed, we check the max(topographic_error, quantization_error) reduces
        data = np.random.rand(1000, 5)
        for neighborhood in ["gaussian", "cut_gaussian", "bubble", "epanechicov"]:
            for normalization in ["standard", "minmax", "log", "logistic", "boxcox"]:
                for distance_metric in ["euclidean", "cityblock"]:
                    for lattice in ["rect", "hexa"]:
                        model = SOM(neighborhood=neighborhood, normalization=normalization, mapsize=[15, 10],
                                    lattice=lattice, distance_metric=distance_metric, n_jobs=1)
                        model.codebook.random_initialization(data)  # Manually initialize the codebook
                        model.data_norm = model.normalizer.normalize(data)
                        e_q, e_t = model.calculate_quantization_error(), model.calculate_topographic_error()
                        max_error_0 = max(e_q, e_t)

                        model.fit(data, epochs=10, radiusin=10, radiusfin=3)
                        e_q, e_t = model.calculate_quantization_error(), model.calculate_topographic_error()
                        max_error_1 = max(e_q, e_t)

                        self.assertGreater(max_error_0, max_error_1)

    def test_find_first_bmus(self):
        data = np.random.rand(1000, 5)
        model = SOM(neighborhood="gaussian", normalization="standard", mapsize=[15, 10], lattice="hexa",
                    distance_metric="euclidean", n_jobs=1)
        model.codebook.random_initialization(data)  # Manually initialize the codebook
        model.data_norm = model.normalizer.normalize(data)

        i = random.sample(range(150), 50)
        input_data = model.codebook.matrix[i, :]

        # Single thread
        bmus = find_bmu(codebook=model.codebook, input_matrix=input_data, metric="euclidean", njb=1)
        self.assertTrue((bmus[0] == i).all())
        self.assertTrue((bmus[1] == 0).all())

        # Multi thread
        bmus = find_bmu(codebook=model.codebook, input_matrix=input_data, metric="euclidean", njb=6)
        self.assertTrue((bmus[0] == i).all())
        self.assertTrue((bmus[1] == 0).all())

    def test_find_second_bmus(self):
        data = np.random.rand(1000, 230)
        model = SOM(neighborhood="gaussian", normalization="standard", mapsize=[15, 10], lattice="hexa",
                    distance_metric="euclidean", n_jobs=1)
        model.codebook.random_initialization(data)  # Manually initialize the codebook
        model.data_norm = model.normalizer.normalize(data)

        i = random.sample(range(148), 50)
        j = [x + 1 for x in i]
        k = [x + 2 for x in i]
        input_data = model.codebook.matrix[i, :]*0.6 + model.codebook.matrix[j, :]*0.4

        # Single thread
        bmus = find_bmu(codebook=model.codebook, input_matrix=input_data, metric="euclidean", njb=1, nth=1)
        self.assertTrue((bmus[0] == i).all())
        bmus = find_bmu(codebook=model.codebook, input_matrix=input_data, metric="euclidean", njb=1, nth=2)
        self.assertTrue((bmus[0] == j).all())

        # Multi thread
        bmus = find_bmu(codebook=model.codebook, input_matrix=input_data, metric="euclidean", njb=6, nth=1)
        self.assertTrue((bmus[0] == i).all())
        bmus = find_bmu(codebook=model.codebook, input_matrix=input_data, metric="euclidean", njb=6, nth=2)
        self.assertTrue((bmus[0] == j).all())
