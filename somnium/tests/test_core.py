from unittest import TestCase
import numpy as np
import random
from scipy.spatial.distance import cdist

from somnium.core import SOM, find_bmu, estimate_mapsize
from somnium.exceptions import ModelNotTrainedError, InvalidValuesInDataSet


class TestGeneralTraining(TestCase):
    def test_fit_rect(self):
        # Check that the training process effectively reduces the errors
        data = np.random.rand(100, 10)
        model = SOM(neighborhood="gaussian", normalization="standard", mapsize=[15, 10], lattice="rect",
                    distance_metric="euclidean", n_jobs=1)
        model.codebook.random_initialization(data)  # Manually initialize the codebook
        model.data_norm = model.normalizer.normalize(data)
        model.model_is_unfitted = False
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
        model.model_is_unfitted = False
        e_q, e_t = model.calculate_quantization_error(), model.calculate_topographic_error()
        f1_0 = 1/(1/e_q + 1/e_t)

        model.fit(data, epochs=10, radiusin=10, radiusfin=3)
        e_q, e_t = model.calculate_quantization_error(), model.calculate_topographic_error()
        f1_1 = 1/(1/e_q + 1/e_t)

        self.assertGreater(f1_0, f1_1)

    def test_predict(self):
        data = np.random.rand(100, 10)
        model = SOM(neighborhood="gaussian", normalization="standard", mapsize=[15, 10], lattice="hexa",
                    distance_metric="euclidean", n_jobs=1)
        model.fit(data, epochs=10, radiusin=10, radiusfin=3)
        bmus_predict = model.predict(data)
        self.assertTrue((model.bmu[0] == bmus_predict).all())

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
                        model.model_is_unfitted = False
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
        model.model_is_unfitted = False

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
        model.model_is_unfitted = False

        i = random.sample(range(148), 50)
        j = [x + 1 for x in i]
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


class TestModelExceptions(TestCase):
    def test_raises_exception_when_model_unfitted(self):
        # Assure an exception is dropped when trying to calculate errors before training
        model = SOM(neighborhood="gaussian", normalization="standard", mapsize=[15, 10], lattice="hexa",
                    distance_metric="euclidean", n_jobs=1)
        self.assertRaises(ModelNotTrainedError, model.calculate_topographic_error)
        self.assertRaises(ModelNotTrainedError, model.calculate_quantization_error)

    def test_nans_catching(self):
        # Assure it drops an error when a NaN value is introduced in the data
        data = np.random.rand(1000, 230)
        data[25, 21] = np.nan
        model = SOM(neighborhood="gaussian", normalization="standard", mapsize=[15, 10], lattice="hexa",
                    distance_metric="euclidean", n_jobs=1)

        self.assertRaises(InvalidValuesInDataSet, model.fit, data=data, epochs=10, radiusin=10, radiusfin=3)

    def test_infs_catching(self):
        # Assure it drops an error when a Inf value is introduced in the data
        data = np.random.rand(1000, 230)
        data[25, 21] = np.inf
        model = SOM(neighborhood="gaussian", normalization="standard", mapsize=[15, 10], lattice="hexa",
                    distance_metric="euclidean", n_jobs=1)
        self.assertRaises(InvalidValuesInDataSet, model.fit, data=data, epochs=10, radiusin=10, radiusfin=3)

        data = np.random.rand(1000, 230)
        data[25, 21] = -np.inf
        model = SOM(neighborhood="gaussian", normalization="standard", mapsize=[15, 10], lattice="hexa",
                    distance_metric="euclidean", n_jobs=1)
        self.assertRaises(InvalidValuesInDataSet, model.fit, data=data, epochs=10, radiusin=10, radiusfin=3)


class TestAutoMapsize(TestCase):
    def test_estimate_mapsize_returns_tuple(self):
        data = np.random.rand(1000, 10)
        ms = estimate_mapsize(data)
        self.assertIsInstance(ms, tuple)
        self.assertEqual(len(ms), 2)
        self.assertGreater(ms[0], 0)
        self.assertGreater(ms[1], 0)

    def test_auto_mapsize_in_som(self):
        data = np.random.rand(500, 5)
        model = SOM(mapsize="auto")
        model.fit(data, epochs=5, radiusin=10, radiusfin=3)
        self.assertGreater(model.codebook.n_rows, 0)
        self.assertGreater(model.codebook.n_columns, 0)

    def test_larger_data_gives_larger_map(self):
        small = estimate_mapsize(np.random.rand(100, 5))
        large = estimate_mapsize(np.random.rand(10000, 5))
        self.assertGreater(large[0] * large[1], small[0] * small[1])


class TestExponentialDecay(TestCase):
    def test_exponential_decay_trains(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(10, 10))
        model.fit(data, epochs=10, radiusin=10, radiusfin=1, decay="exponential")
        qe = model.calculate_quantization_error()
        self.assertGreater(qe, 0)


class TestFitAuto(TestCase):
    def test_fit_auto_trains(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(10, 10))
        model.fit_auto(data, rough_epochs=10, fine_epochs=10)
        qe = model.calculate_quantization_error()
        te = model.calculate_topographic_error()
        self.assertGreater(qe, 0)
        self.assertGreater(te, 0)

    def test_fit_auto_with_auto_mapsize(self):
        data = np.random.rand(500, 5)
        model = SOM(mapsize="auto")
        model.fit_auto(data, rough_epochs=10, fine_epochs=10)
        self.assertGreater(model.codebook.n_rows, 0)

    def test_fit_auto_reduces_error(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(8, 8), initialization="pca")
        model.fit_auto(data, rough_epochs=20, fine_epochs=20)
        qe = model.calculate_quantization_error()
        te = model.calculate_topographic_error()
        # Should achieve reasonable metrics
        self.assertLess(qe, 0.5)
        self.assertLess(te, 0.3)


class TestVacancyRate(TestCase):
    def test_vacancy_rate_range(self):
        data = np.random.rand(500, 5)
        model = SOM(mapsize=(10, 10))
        model.fit(data, epochs=10, radiusin=10, radiusfin=3)
        vr = model.calculate_vacancy_rate()
        self.assertGreaterEqual(vr, 0.0)
        self.assertLessEqual(vr, 1.0)

    def test_vacancy_rate_unfitted_raises(self):
        model = SOM(mapsize=(10, 10))
        self.assertRaises(ModelNotTrainedError, model.calculate_vacancy_rate)


class TestPCAInitialization(TestCase):
    def test_pca_init_trains(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(10, 10), initialization="pca")
        model.fit(data, epochs=10, radiusin=10, radiusfin=3)
        qe = model.calculate_quantization_error()
        self.assertGreater(qe, 0)

    def test_invalid_initialization_raises(self):
        self.assertRaises(ValueError, SOM, initialization="invalid")


class TestToroidalTraining(TestCase):
    def test_toroidal_hexa_trains(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(10, 10), lattice="toroidal_hexa")
        model.fit(data, epochs=10, radiusin=10, radiusfin=3)
        self.assertGreater(model.calculate_quantization_error(), 0)

    def test_cylindrical_rect_trains(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(10, 10), lattice="cylindrical_rect")
        model.fit(data, epochs=10, radiusin=10, radiusfin=3)
        self.assertGreater(model.calculate_quantization_error(), 0)


class TestLearningRate(TestCase):
    def test_learning_rate_trains(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(10, 10))
        model.fit(data, epochs=10, radiusin=10, radiusfin=3, learning_rate=0.5)
        self.assertGreater(model.calculate_quantization_error(), 0)

    def test_learning_rate_below_1_smooths(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(10, 10), initialization="pca")
        model.fit(data, epochs=20, radiusin=10, radiusfin=3, learning_rate=0.5)
        qe = model.calculate_quantization_error()
        # Should still converge to a reasonable QE
        self.assertGreater(qe, 0)
        self.assertLess(qe, 1)


class TestSubsampleRatio(TestCase):
    def test_subsample_trains(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(10, 10))
        model.fit(data, epochs=10, radiusin=10, radiusfin=3, subsample_ratio=0.5)
        qe = model.calculate_quantization_error()
        self.assertGreater(qe, 0)

    def test_subsample_with_learning_rate(self):
        data = np.random.rand(50, 5)
        model = SOM(mapsize=(10, 10))
        model.fit(data, epochs=20, radiusin=10, radiusfin=3,
                  subsample_ratio=0.8, learning_rate=0.5)
        vr = model.calculate_vacancy_rate()
        self.assertGreaterEqual(vr, 0)
        self.assertLessEqual(vr, 1)


class TestCollectHistory(TestCase):
    def test_history_has_correct_keys_and_length(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(8, 8))
        model.fit(data, epochs=10, radiusin=8, radiusfin=2, collect_history=True)
        self.assertTrue(hasattr(model, 'history_'))
        for key in ("quantization_error", "topographic_error", "vacancy_rate"):
            self.assertIn(key, model.history_)
            self.assertEqual(len(model.history_[key]), 10)

    def test_history_values_are_valid(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(8, 8))
        model.fit(data, epochs=10, radiusin=8, radiusfin=2, collect_history=True)
        for qe in model.history_["quantization_error"]:
            self.assertGreater(qe, 0)
        for te in model.history_["topographic_error"]:
            self.assertGreaterEqual(te, 0)
            self.assertLessEqual(te, 1)
        for vr in model.history_["vacancy_rate"]:
            self.assertGreaterEqual(vr, 0)
            self.assertLessEqual(vr, 1)

    def test_qe_decreases_over_training(self):
        np.random.seed(42)
        data = np.random.rand(300, 5)
        model = SOM(mapsize=(8, 8))
        model.fit(data, epochs=30, radiusin=8, radiusfin=1, collect_history=True)
        qe = model.history_["quantization_error"]
        # First third average should be higher than last third average
        self.assertGreater(np.mean(qe[:10]), np.mean(qe[-10:]))

    def test_history_accumulates_across_fit_calls(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(8, 8))
        model.fit(data, epochs=5, radiusin=8, radiusfin=3, collect_history=True)
        model.fit(data, epochs=5, radiusin=3, radiusfin=1, collect_history=True)
        self.assertEqual(len(model.history_["quantization_error"]), 10)

    def test_no_history_by_default(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(8, 8))
        model.fit(data, epochs=5, radiusin=8, radiusfin=2)
        self.assertFalse(hasattr(model, 'history_'))

    def test_history_with_exponential_decay(self):
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(8, 8))
        model.fit(data, epochs=10, radiusin=8, radiusfin=1,
                  decay="exponential", collect_history=True)
        self.assertEqual(len(model.history_["quantization_error"]), 10)


class TestTopographicErrorCorrectness(TestCase):
    def _te_reference(self, model):
        """Reference TE implementation using the naive per-point Python loop."""
        metric = model.codebook.lattice.distance_metric
        d = cdist(model.codebook.matrix, model.data_norm, metric=metric)
        bmu1 = np.argpartition(d, 0, axis=0)[0]
        bmu2 = np.argpartition(d, 1, axis=0)[1]
        lattice = model.codebook.lattice
        neighs = [lattice.are_neighbor_indices(int(b1), int(b2))
                  for b1, b2 in zip(bmu1, bmu2)]
        return 1 - np.mean(neighs)

    def test_te_matches_reference_hexa(self):
        np.random.seed(123)
        data = np.random.rand(300, 5)
        model = SOM(mapsize=(10, 10), lattice="hexa")
        model.fit(data, epochs=15, radiusin=10, radiusfin=1)
        te_model = model.calculate_topographic_error()
        te_ref = self._te_reference(model)
        self.assertAlmostEqual(te_model, te_ref, places=10)

    def test_te_matches_reference_rect(self):
        np.random.seed(456)
        data = np.random.rand(300, 5)
        model = SOM(mapsize=(10, 10), lattice="rect")
        model.fit(data, epochs=15, radiusin=10, radiusfin=1)
        te_model = model.calculate_topographic_error()
        te_ref = self._te_reference(model)
        self.assertAlmostEqual(te_model, te_ref, places=10)

    def test_te_matches_reference_toroidal_hexa(self):
        np.random.seed(789)
        data = np.random.rand(300, 5)
        model = SOM(mapsize=(8, 8), lattice="toroidal_hexa")
        model.fit(data, epochs=15, radiusin=8, radiusfin=1)
        te_model = model.calculate_topographic_error()
        te_ref = self._te_reference(model)
        self.assertAlmostEqual(te_model, te_ref, places=10)

    def test_te_matches_reference_cylindrical_rect(self):
        np.random.seed(101)
        data = np.random.rand(300, 5)
        model = SOM(mapsize=(8, 8), lattice="cylindrical_rect")
        model.fit(data, epochs=15, radiusin=8, radiusfin=1)
        te_model = model.calculate_topographic_error()
        te_ref = self._te_reference(model)
        self.assertAlmostEqual(te_model, te_ref, places=10)

    def test_te_history_matches_model_te(self):
        np.random.seed(42)
        data = np.random.rand(200, 5)
        model = SOM(mapsize=(8, 8))
        model.fit(data, epochs=5, radiusin=8, radiusfin=2, collect_history=True)
        # Last history TE should match model.calculate_topographic_error()
        # (both computed on the full data after the last codebook update)
        te_history_last = model.history_["topographic_error"][-1]
        te_model = model.calculate_topographic_error()
        self.assertAlmostEqual(te_history_last, te_model, places=10)
