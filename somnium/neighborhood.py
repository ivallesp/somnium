import numpy as np
import inspect
import sys

epsilon = .000000000001


class NeighborhoodFactory(object):
    """
    Factory class for neighborhood functions
    """
    @staticmethod
    def build(neighborhood_func):
        """
        Looks for the existing neighborhood functions and returns the chosen one if exists. If it doesn't, it throws
        an exception.
        :param neighborhood_func: name of the neighborhood function
        :return: lattice class (class)
        """
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and neighborhood_func == obj.name:
                    return obj()
        else:
            raise Exception(
                "Unsupported neighborhood function '%s'" % neighborhood_func)


class GaussianNeighborhood:
    """
    Gaussian neighborhood function

    f(x) = e^{-\frac{x^2}{2*r^2}}

    """
    name = 'gaussian'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        """
        Given a distance matrix, it applies the neighborhood function
        :param distance_matrix: matrix with the pairwise distances (np.array)
        :param radius: radius for the neighborhood function (float)
        :param dim: number of dimensions in the distance matrix (int)
        :return: distance matrix with the neighborhood function applied (np.array)
        """
        return np.exp(-(distance_matrix**2)/(2.0*radius**2)).reshape(dim, dim)

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)


class BubbleNeighborhood:
    """
    Bubble neighborhood function

    f(x) = 1 \text{if} x <= r
    f(x) = 0 \text{if} x > r

    """
    name = 'bubble'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        """
        Given a distance matrix, it applies the neighborhood function
        :param distance_matrix: matrix with the pairwise distances (np.array)
        :param radius: radius for the neighborhood function (float)
        :param dim: number of dimensions in the distance matrix (int)
        :return: distance matrix with the neighborhood function applied (np.array)
        """
        return np.where(distance_matrix > radius, 0.0, 1.0).reshape(dim, dim) + epsilon

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)


class CutGaussianNeighborhood:
    """
    Bubble neighborhood function

    f(x) = e^{-\frac{x^2}{2*r^2}} \text{if} x <= r
    f(x) = 0 \text{if} x > r

    """
    name = 'cut_gaussian'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        """
        Given a distance matrix, it applies the neighborhood function
        :param distance_matrix: matrix with the pairwise distances (np.array)
        :param radius: radius for the neighborhood function (float)
        :param dim: number of dimensions in the distance matrix (int)
        :return: distance matrix with the neighborhood function applied (np.array)
        """
        gaussian = np.exp(-(distance_matrix**2)/(2.0*radius**2)).reshape(dim, dim)
        threshold = np.exp(-(radius**2)/(2.0*radius**2))
        gaussian[gaussian < threshold] = epsilon  # Cut at radius
        return gaussian

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)


class EpanechicovNeighborhood:
    """
    Epanechicov neighborhood function

    f(x) = max(1-(x/r)^2, 0)

    """
    name = 'epanechicov'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        """
        Given a distance matrix, it applies the neighborhood function
        :param distance_matrix: matrix with the pairwise distances (np.array)
        :param radius: radius for the neighborhood function (float)
        :param dim: number of dimensions in the distance matrix (int)
        :return: distance matrix with the neighborhood function applied (np.array)
        """
        return np.clip(1 - (distance_matrix/radius) ** 2, epsilon, None).reshape(dim, dim)

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)