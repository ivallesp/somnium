# _Somnium_: flexible Self-Organising Maps implementation
[![Build Status](https://travis-ci.com/ivallesp/somnium.svg?branch=master)](https://travis-ci.com/ivallesp/somnium)
[![Code coverage](https://codecov.io/gh/ivallesp/somnium/branch/master/graph/badge.svg)](https://codecov.io/gh/ivallesp/somnium)

## What is it?
Somnium is a library intended for providing a easy and powerful way of exploring multi-dimensional data sets. It uses the Self-Organising Map algorithm (aka Kohonen map).
 
 A _Self-Organising Map_ (_SOM_ hereafter), is a biologically inspired algorithm meant for exploring multi-dimensional non-linear relations between variables. SOM was proposed in 1984 by Teuvo Kohonen, a Finnish academician. It is based in the process of task clustering that occurs in our brain and it is considered a type of neural network. It compresses the information of high-dimensional data into geometric relationships onto a low-dimensional representation.

## Main aplications
Here are just a few of the applications of SOM algorithm.

- Discover, at a glance, the non-linear relations between the variables of a dataset.
- Micro-segment the instances of a dataset in an easily and visually understandable way.
- Work as a surrogate-model of a black-box model for explainability purpose.
- Assistant for feature selection/reduction by finding linearly and non-linearly correlated variables.

## Installation
### Dependencies
Somnium requires:
- Python (>=3.5)
- NumPy (>= 1.7)
- SciPy (>= 1.1)
- scikit-learn (>= 0.20)

### User installation
For now, the only supported installation method is through `setuptools`:
1. Clone the repository pasting the following code in your terminal.
    ```
    git clone https://github.com/ivallesp/somnium
    ```
2. Move your current directory to the main folder in the repository:
    ```
    cd somnium
    ```
3. Install the package using python:
    ```
    python setup.py install
    ```

## How to use it
The API is currently being developed, which means that it is going to change from time to time. However the `master` branch of this repository will always be fully functional. In the future, I plan to write some docs about the library, but for now you can find at least one example of usage in the `examples` folder.

## Development lines
- Integrate the visualization into the _SOM_ _API_.
- Enhance the visualization _API_ with more OOP patterns.
- Write a documentation page
- Integrate with _Travis_.
- Work on other installation methods.
- Write at least one example for each application.
- Research for and implement algorithm enhancements.
- Enhance reproducibility. Start by setting a seed.
- Refactor plugins for always returning the figure, no `plt.show()

## Known issues
- The current visualization engine only runs well under ``jupyter notebooks``. If you run it from a _python_ or _ipython_ console the figures will not look well.
- Wider than higher maps (e.g. `mapsize=[10, 15]`) are not shown correctly.

## Contributing
All contributions are welcome and appreciated. I don't have time to finish it soon so, please, feel free to open an issue to either propose some contribution or discuss potential new functionalities. All the contributions should be made through a _pull request_. 

## Attribution
This library has been built using [SOMPY](https://github.com/sevamoo/SOMPY) library as a starting point, and that is why you may find some similarities in the code.

## License
This library has been licensed under MIT agreement. Please refer to the `LICENSE` file on the root of this repository. Copyright (c) 2019 Iván Vallés Pérez