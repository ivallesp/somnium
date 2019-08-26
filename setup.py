try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


setup(
    name="SOMNIUM",
    version="0",
    description="A versatile Self Organising Maps implementation in Python",
    author="Iván Vallés Pérez",
    author_email="ivanvallesperez@gmail.com",
    packages=find_packages(),
    install_requires=['numpy >= 1.7', 'scipy >= 1.1', 'scikit-learn >= 0.20']
)
