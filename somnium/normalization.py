import numpy as np
import sys
import inspect
from scipy.stats import boxcox
from scipy.special import inv_boxcox


class NormalizerFactory(object):
    """
    Factory class used to retrieve the supported normalizers
    """
    @staticmethod
    def build(name):
        """
        Looks for the existing normalizers and returns the chosen one if exists. If it doesn't, it throws an exception
        :param name: name of the normalizer
        :return: normalizer function (func)
        """
        for _, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and name == obj.name:
                    return obj()
        else:
            raise Exception("Unknown normalization type '%s'" % name)


class Normalizer(object):
    """
    Abstract class, prototype of normalizer classes.
    """
    def normalize(self, data):
        raise NotImplementedError()

    def denormalize(self, data):
        raise NotImplementedError()

    def normalize_by(self, raw_data, data):
        raise NotImplementedError()

    def denormalize_by(self, raw_data, data):
        raise NotImplementedError()


class StandardNormalizer(Normalizer):
    """
    Standard scaling and centering normalization
    z = \frac{x - mean(x)}{std(x)}
    """
    name = 'standard'

    def _mean_and_standard_dev(self, data):
        """
        Private method for calculating the mean and std dev. These statistics are calculated over the 0th axis.
        :param data: data to use to calculate the stats (np.array)
        :return: mean and std (np.arrays)
        """
        return np.mean(data, axis=0), np.std(data, axis=0)

    def normalize(self, data):
        """
        Calculates the statistics over the data matrix, stores them in attributes and then apply them to the data
        :param data: data to normalize (np.array)
        :return: normalized data (np.array)
        """
        self.me, self.st = self._mean_and_standard_dev(data)
        self.st[self.st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        return (data-self.me)/self.st

    def denormalize(self, data):
        """
        Undo the normalization with the stored parameters
        :param data: data to denormalize (np.array)
        :return:
        """
        return data*self.st + self.me

    def normalize_by(self, raw_data, data):
        """
        Normalize using stats calculated over an external data set
        :param raw_data: external data set used to calculate the statistics (np.array)
        :param data: data to normalize (np.array)
        :return: normalized data (np.array)
        """
        me, st = self._mean_and_standard_dev(raw_data)
        st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        return (data-me)/st

    def denormalize_by(self, raw_data, data):
        """
        Undo the normalization using stats calculated over an external data set
        :param raw_data: external data set used to calculate the statistics (np.array)
        :param data: data to denormalize (np.array)
        :return: denormalized data (np.array)
        """
        me, st = self._mean_and_standard_dev(raw_data)
        return data * st + me


class MinMaxNormalizer(Normalizer):
    """
    Min-Max centering and scaling normalization
    z = \frac{x - min(x)}{max(x)}
    """
    name = 'minmax'
    def _min_and_max(self, data):
        """
        Private method for calculating the min and max. These statistics are calculated over the 0th axis.
        :param data: data to use to calculate the stats (np.array)
        :return: mean and std (np.arrays)
        """
        return np.min(data, axis=0), np.max(data, axis=0)

    def normalize(self, data):
        """
        Calculates the statistics over the data matrix, stores them in attributes and then apply them to the data
        :param data: data to normalize (np.array)
        :return: normalized data (np.array)
        """
        self.min, self.max = self._min_and_max(data)
        return (data-self.min)/self.max

    def denormalize(self, data):
        """
        Undo the normalization with the stored parameters
        :param data: data to denormalize (np.array)
        :return:
        """
        return data*self.max + self.min

    def normalize_by(self, raw_data, data):
        """
        Normalize using stats calculated over an external data set
        :param raw_data: external data set used to calculate the statistics (np.array)
        :param data: data to normalize (np.array)
        :return: normalized data (np.array)
        """
        minimum, max = self._min_and_max(raw_data)
        return (data-minimum)/max

    def denormalize_by(self, raw_data, data):
        """
        Undo the normalization using stats calculated over an external data set
        :param raw_data: external data set used to calculate the statistics (np.array)
        :param data: data to denormalize (np.array)
        :return: denormalized data (np.array)
        """
        minimum, max = self._min_and_max(raw_data)
        return data * max + minimum


class LogNormalizer(Normalizer):
    """
    Logarithm normalization
    z = log(x+1)
    """
    name = 'log'
    def normalize(self, data):
        """
        Calculates the statistics over the data matrix, stores them in attributes and then apply them to the data
        :param data: data to normalize (np.array)
        :return: normalized data (np.array)
        """
        return np.log1p(data)

    def denormalize(self, data):
        """
        Undo the normalization with the stored parameters
        :param data: data to denormalize (np.array)
        :return:
        """
        return np.expm1(data)

    def normalize_by(self, raw_data, data):
        """
        Normalize using stats calculated over an external data set
        :param raw_data: external data set used to calculate the statistics (np.array)
        :param data: data to normalize (np.array)
        :return: normalized data (np.array)
        """
        return np.log1p(data)

    def denormalize_by(self, raw_data, data):
        """
        Undo the normalization using stats calculated over an external data set
        :param raw_data: external data set used to calculate the statistics (np.array)
        :param data: data to denormalize (np.array)
        :return: denormalized data (np.array)
        """
        return np.expm1(data)


class LogisticNormalizer(Normalizer):
    """
    Logistic function normalization, after a standard normalization.
     x' = \frac{x - mean(x)}{std(x)}
     z = log(x')
    """
    name = 'logistic'
    def normalize(self, data):
        """
        Calculates the statistics over the data matrix, stores them in attributes and then apply them to the data
        :param data: data to normalize (np.array)
        :return: normalized data (np.array)
        """
        self.stdsc = VarianceNormalizer()
        data = self.stdsc.normalize(data)
        return 1/(1+np.exp(-data))

    def denormalize(self, data):
        """
        Undo the normalization with the stored parameters
        :param data: data to denormalize (np.array)
        :return:
        """
        data = np.log(data/(1-data))
        data = self.stdsc.denormalize(data)
        return data

    def normalize_by(self, raw_data, data):
        """
        Normalize using stats calculated over an external data set
        :param raw_data: external data set used to calculate the statistics (np.array)
        :param data: data to normalize (np.array)
        :return: normalized data (np.array)
        """
        data = VarianceNormalizer().normalize_by(raw_data, data)
        return 1/(1+np.exp(-data))

    def denormalize_by(self, raw_data, data):
        """
        Undo the normalization using stats calculated over an external data set
        :param raw_data: external data set used to calculate the statistics (np.array)
        :param data: data to denormalize (np.array)
        :return: denormalized data (np.array)
        """
        data = np.log(data/(1-data))
        data = VarianceNormalizer().denormalize_by(raw_data, data)
        return data


class BoxCox:
    """
    Box-Cox function normalization, after a standard normalization.
     x' = \frac{x - mean(x)}{std(x)}
     z = box-cox(x)
    """
    name = 'boxcox'
    @staticmethod
    def _boxcox(data, lambdas=None):
        """
        Performs the boxcox transformation over the data and stores lambdas in order to inverse_transform when needed
        :param data: data to transform (np.array)
        :param lambdas: lambdas to use, if not specified, they are automatically inferred (list or iterable or None)
        :return: data normalized and lambdas (tuple)
        """
        data, lambdas = list(zip(*map(boxcox, data.T))) if lambdas is None else list(zip(*map(boxcox, data.T, lambdas)))
        return np.array(data).T, lambdas

    @staticmethod
    def _inv_boxcox(data, lambdas):
        """
        Performs the inverse boxcox transformation over the data using the specified lambdas
        :param data: data to inverse transform (np.array)
        :param lambdas: lambdas to use (list or iterable)
        :return: data denormalized and lambdas (tuple)
        """
        data = list(zip(*map(inv_boxcox, data.T, lambdas)))
        return np.array(data)

    def normalize(self, data):
        """
        Calculates the statistics over the data matrix, stores them in attributes and then apply them to the data
        :param data: data to normalize (np.array)
        :return: normalized data (np.array)
        """
        self.stdnorm = VarianceNormalizer()
        data, self.lambdas = self._boxcox(data)
        data = self.stdnorm.normalize(data)
        return data

    def denormalize(self, data):
        """
        Undo the normalization with the stored parameters
        :param data: data to denormalize (np.array)
        :return:
        """
        data = self.stdnorm.denormalize(data)
        data = self._inv_boxcox(data, self.lambdas)
        return data

    def normalize_by(self, raw_data, data):
        """
        Normalize using stats calculated over an external data set
        :param raw_data: external data set used to calculate the statistics (np.array)
        :param data: data to normalize (np.array)
        :return: normalized data (np.array)
        """
        stdnorm = VarianceNormalizer()
        raw_data_norm, lambdas = self._boxcox(raw_data)
        data, _ = self._boxcox(data, lambdas)
        data = stdnorm.normalize_by(raw_data_norm, data)
        return data

    def denormalize_by(self, raw_data, data):
        """
        Undo the normalization using stats calculated over an external data set
        :param raw_data: external data set used to calculate the statistics (np.array)
        :param data: data to denormalize (np.array)
        :return: denormalized data (np.array)
        """
        stdnorm = VarianceNormalizer()
        raw_data_norm, lambdas = self._boxcox(raw_data)
        data = stdnorm.denormalize_by(raw_data_norm, data)
        data = self._inv_boxcox(data, lambdas)
        return data
