from unittest import TestCase
import numpy as np

from somnium.normalization import NormalizerFactory, Normalizer
from somnium.exceptions import NormalizationFunctionNotFound

class TestNormalization(TestCase):
    def setUp(self):
        self.data = np.random.rand(1000, 10) * 7  # Mock data
        self.data_aux = np.random.rand(1000, 10) * 7  # Mock data

    def test_standard_normalizer(self):
        normalizer = NormalizerFactory.build("standard")
        data_norm = normalizer.normalize(self.data)
        data_denorm = normalizer.denormalize(data_norm)
        data_norm_by = normalizer.normalize_by(self.data_aux, self.data)
        data_denorm_by = normalizer.denormalize_by(self.data_aux, data_norm_by)

        # The input and output matrices have the same shape
        self.assertEqual(self.data.shape, data_norm.shape)
        self.assertEqual(self.data.shape, data_denorm.shape)
        self.assertEqual(self.data.shape, data_norm_by.shape)
        self.assertEqual(self.data.shape, data_denorm_by.shape)

        # The normalized data is NOT the same as the input data
        self.assertTrue(np.sometrue(data_norm != data_denorm))
        self.assertTrue(np.sometrue(data_norm_by != data_denorm_by))

        # The denormalized data is the same as the input data
        self.assertTrue(np.allclose(self.data, data_denorm))
        self.assertTrue(np.allclose(self.data, data_denorm_by))

        # The normalized_by and the normalized data are not the same
        self.assertTrue(np.sometrue(data_norm != data_norm_by))

    def test_minmax_normalizer(self):
        normalizer = NormalizerFactory.build("minmax")
        data_norm = normalizer.normalize(self.data)
        data_denorm = normalizer.denormalize(data_norm)
        data_norm_by = normalizer.normalize_by(self.data_aux, self.data)
        data_denorm_by = normalizer.denormalize_by(self.data_aux, data_norm_by)

        # The input and output matrices have the same shape
        self.assertEqual(self.data.shape, data_norm.shape)
        self.assertEqual(self.data.shape, data_denorm.shape)
        self.assertEqual(self.data.shape, data_norm_by.shape)
        self.assertEqual(self.data.shape, data_denorm_by.shape)

        # The normalized data is NOT the same as the input data
        self.assertTrue(np.sometrue(data_norm != data_denorm))
        self.assertTrue(np.sometrue(data_norm_by != data_denorm_by))

        # The denormalized data is the same as the input data
        self.assertTrue(np.allclose(self.data, data_denorm))
        self.assertTrue(np.allclose(self.data, data_denorm_by))

        # The normalized_by and the normalized data are not the same
        self.assertTrue(np.sometrue(data_norm != data_norm_by))

    def test_log_normalizer(self):
        normalizer = NormalizerFactory.build("log")
        data_norm = normalizer.normalize(self.data)
        data_denorm = normalizer.denormalize(data_norm)
        data_norm_by = normalizer.normalize_by(self.data_aux, self.data)
        data_denorm_by = normalizer.denormalize_by(self.data_aux, data_norm_by)

        # The input and output matrices have the same shape
        self.assertEqual(self.data.shape, data_norm.shape)
        self.assertEqual(self.data.shape, data_denorm.shape)
        self.assertEqual(self.data.shape, data_norm_by.shape)
        self.assertEqual(self.data.shape, data_denorm_by.shape)

        # The normalized data is NOT the same as the input data
        self.assertTrue(np.sometrue(data_norm != data_denorm))
        self.assertTrue(np.sometrue(data_norm_by != data_denorm_by))

        # The denormalized data is the same as the input data
        self.assertTrue(np.allclose(self.data, data_denorm))
        self.assertTrue(np.allclose(self.data, data_denorm_by))

        # The normalized_by and the normalized data ARE the same
        self.assertTrue(np.alltrue(data_norm == data_norm_by))

    def test_logistic_normalizer(self):
        normalizer = NormalizerFactory.build("logistic")
        data_norm = normalizer.normalize(self.data)
        data_denorm = normalizer.denormalize(data_norm)
        data_norm_by = normalizer.normalize_by(self.data_aux, self.data)
        data_denorm_by = normalizer.denormalize_by(self.data_aux, data_norm_by)

        # The input and output matrices have the same shape
        self.assertEqual(self.data.shape, data_norm.shape)
        self.assertEqual(self.data.shape, data_denorm.shape)
        self.assertEqual(self.data.shape, data_norm_by.shape)
        self.assertEqual(self.data.shape, data_denorm_by.shape)

        # The normalized data is NOT the same as the input data
        self.assertTrue(np.sometrue(data_norm != data_denorm))
        self.assertTrue(np.sometrue(data_norm_by != data_denorm_by))

        # The denormalized data is the same as the input data
        self.assertTrue(np.allclose(self.data, data_denorm))
        self.assertTrue(np.allclose(self.data, data_denorm_by))

        # The normalized_by and the normalized data are not the same
        self.assertTrue(np.sometrue(data_norm != data_norm_by))

    def test_boxcox_normalizer(self):
        normalizer = NormalizerFactory.build("boxcox")
        data_norm = normalizer.normalize(self.data)
        data_denorm = normalizer.denormalize(data_norm)
        data_norm_by = normalizer.normalize_by(self.data_aux, self.data)
        data_denorm_by = normalizer.denormalize_by(self.data_aux, data_norm_by)

        # The input and output matrices have the same shape
        self.assertEqual(self.data.shape, data_norm.shape)
        self.assertEqual(self.data.shape, data_denorm.shape)
        self.assertEqual(self.data.shape, data_norm_by.shape)
        self.assertEqual(self.data.shape, data_denorm_by.shape)

        # The normalized data is NOT the same as the input data
        self.assertTrue(np.sometrue(data_norm != data_denorm))
        self.assertTrue(np.sometrue(data_norm_by != data_denorm_by))

        # The denormalized data is the same as the input data
        self.assertTrue(np.allclose(self.data, data_denorm))
        self.assertTrue(np.allclose(self.data, data_denorm_by))

        # The normalized_by and the normalized data are not the same
        self.assertTrue(np.sometrue(data_norm != data_norm_by))

class TestNeighborhoodExceptions(TestCase):
    def test_unrecognized_neighborhood_exception(self):
        # Assure an exception is dropped when trying to calculate errors before training
        self.assertRaises(NormalizationFunctionNotFound, NormalizerFactory.build, "foo")
    def test_not_implemented_errors(self):
        # Assure NotImplementedError is raised when a class does not have the proper methods defined.
        class FooNormalizer(Normalizer):
            pass
        data = np.random.rand(1000, 10) * 7  # Mock data
        data_aux = np.random.rand(1000, 10) * 7  # Mock data

        self.assertRaises(NotImplementedError, FooNormalizer().normalize, data=data)
        self.assertRaises(NotImplementedError, FooNormalizer().denormalize, data=data)
        self.assertRaises(NotImplementedError, FooNormalizer().normalize_by, data=data, raw_data=data_aux)
        self.assertRaises(NotImplementedError, FooNormalizer().denormalize_by, data=data, raw_data=data_aux)


