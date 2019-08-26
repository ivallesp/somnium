import numpy as np
import inspect
import sys

small = .000000000001


class NeighborhoodFactory(object):

    @staticmethod
    def build(neighborhood_func):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and neighborhood_func == obj.name:
                    return obj()
        else:
            raise Exception(
                "Unsupported neighborhood function '%s'" % neighborhood_func)


class GaussianNeighborhood(object):
    name = 'gaussian'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        return np.exp(-(distance_matrix**2)/(2.0*radius**2)).reshape(dim, dim)

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)


class BubbleNeighborhood(object):
    name = 'bubble'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        return np.where(distance_matrix > radius, 0.0, 1.0).reshape(dim, dim) + small

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)


class CutGaussianNeighborhood(object):
    name = 'cut_gaussian'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        gaussian = np.exp(-(distance_matrix**2)/(2.0*radius**2)).reshape(dim, dim)
        threshold = np.exp(-(radius**2)/(2.0*radius**2))
        gaussian[gaussian < threshold] = small  # Cut at radius
        return gaussian

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)


class EpanechicovNeighborhood(object):

    name = 'epanechicov'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        return np.clip(1-(distance_matrix/radius)**2, small, None).reshape(dim, dim)

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)