"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from abc import ABC, abstractmethod
import numpy as np


class IGeneratorRandom(ABC):
    """Interface (abstract base class) for wrapping numpy random number
    generation to allow probability distribution parameters to be specified in
    the initialiser whereas the generation of random numbers has a common
    method that just takes the require shape to return. Allows for easy
    subsitution of different probability distributions.
    """

    @abstractmethod
    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Abstract method. Generates an array of random numbers with a shape
        based on the specified shape.

        Parameters
        ----------
        shape : tuple[int,...]
            Shape of the array of random numbers to be returns.

        Returns
        -------
        np.ndarray
            Array of random numbers with shape specified by the input shape.
        """
        pass


class GeneratorNormal(IGeneratorRandom):
    """Class wrapping the numpy normal random number generator. Implements the
    IGeneratorRandom interface to allow for interchangeability with other random
    number generators.
    """
    __slots__ = ("_std","_mean","_rng")

    def __init__(self,
                 std: float = 1.0,
                 mean: float = 0.0,
                 seed: int | None = None) -> None:
        """Initialiser taking the parameters of the probability distribution and
        an optional seed for the random generator to allow for reproducibility.

        Parameters
        ----------
        std : float, optional
            _description_, by default 1.0
        mean : float, optional
            _description_, by default 0.0
        seed : int | None, optional
            _description_, by default None
        """
        self._std =std
        self._mean = mean
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """Initialiser taking the parameters of the probability distribution and
        an optional seed for the random generator to allow for reproducibility.

        Parameters
        ----------
        shape : tuple[int,...]
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        return self._rng.normal(loc = self._mean,
                                scale = self._std,
                                shape = shape)


class GeneratorLogNormal(IGeneratorRandom):
    """Class wrapping the numpy lognormal random generator. Implements the
    IGeneratorRandom interface to allow for interchangeability with other random
    number generators.
    """
    __slots__ = ("_std","_mean","_rng")

    def __init__(self,
                 std: float = 1.0,
                 mean: float = 0.0,
                 seed: int | None = None) -> None:
        """Initialiser taking the parameters of the probability distribution and
        an optional seed for the random generator to allow for reproducibility.

        Parameters
        ----------
        std : float, optional
            _description_, by default 1.0
        mean : float, optional
            _description_, by default 0.0
        seed : int | None, optional
            _description_, by default None
        """
        self._std =std
        self._mean = mean
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        shape : tuple[int,...]
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        return self._rng.lognormal(mean = self._mean,
                                   sigma = self._std,
                                   shape = shape)


class GeneratorUniform(IGeneratorRandom):
    """Class wrapping the numpy uniform random number generator. Implements the
    IGeneratorRandom interface to allow for interchangeability with other random
    number generators.
    """
    __slots__ = ("_low","_high","_rng")

    def __init__(self,
                 low: float = -1.0,
                 high: float = 1.0,
                 seed: int | None = None) -> None:
        """Initialiser taking the parameters of the probability distribution and
        an optional seed for the random generator to allow for reproducibility.

        Parameters
        ----------
        low : float, optional
            _description_, by default -1.0
        high : float, optional
            _description_, by default 1.0
        seed : int | None, optional
            _description_, by default None
        """

        self._low = low
        self._high = high
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        return self._rng.uniform(low = self._low,
                                 high = self._high,
                                 shape = shape)


class GeneratorExponential(IGeneratorRandom):
    """_summary_

    """
    __slots__ = ("_scale","_rng")

    def __init__(self,
                 scale: float = 1.0,
                 seed: int | None = None) -> None:
        """Initialiser taking the parameters of the probability distribution and
        an optional seed for the random generator to allow for reproducibility.

        Parameters
        ----------
        scale : float, optional
            _description_, by default 1.0
        seed : int | None, optional
            _description_, by default None
        """
        self._scale = scale
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        shape : tuple[int,...]
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        return self._rng.exponential(scale = self._scale,
                                     shape = shape)


class GeneratorChiSquare(IGeneratorRandom):
    """_summary_
    """
    __slots__ = ("_dofs","_rng")

    def __init__(self,
                 dofs: float,
                 seed: int | None = None) -> None:
        """Initialiser taking the parameters of the probability distribution and
        an optional seed for the random generator to allow for reproducibility.

        Parameters
        ----------
        dofs : float
            _description_
        seed : int | None, optional
            _description_, by default None
        """
        self._dofs = np.abs(dofs)
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        shape : tuple[int,...]
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        return self._rng.chisquare(df = self._dofs,
                                   shape = shape)


class GeneratorDirichlet(IGeneratorRandom):

    __slots__ = ("_alpha","_rng")

    def __init__(self,
                 alpha: float,
                 seed: int | None = None) -> None:

        self._alpha = alpha
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        return self._rng.dirichlet(alpha = self._alpha, shape = shape)


class GeneratorF(IGeneratorRandom):

    __slots__ = ("_dofs","_rng")

    def __init__(self,
                 dofs: float,
                 seed: int | None = None) -> None:

        self._dofs = np.abs(dofs)
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        return self._rng.f(dfnum = self._dofs, shape = shape)


class GeneratorGamma(IGeneratorRandom):

    __slots__ = ("_shape","_scale","_rng")

    def __init__(self,
                 shape: float,
                 scale: float = 1.0,
                 seed: int | None = None) -> None:

        self._shape = np.abs(shape)
        self._scale = scale
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        return self._rng.gamma(scale = self._scale,
                                     shape = shape)


class GeneratorStudentT(IGeneratorRandom):

    __slots__ = ("_dofs","_rng")

    def __init__(self,
                 dofs: float,
                 seed: int | None = None) -> None:

        self._dofs = np.abs(dofs)
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        return self._rng.standard_t(df = self._dofs,
                                   shape = shape)


class GeneratorBeta(IGeneratorRandom):

    __slots__ = ("_a","_b","_rng")

    def __init__(self,
                 a: float,
                 b: float,
                 seed: int | None = None) -> None:

        self._a = np.abs(a)
        self._b = np.abs(b)
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        return self._rng.beta(a = self._a,
                              b = self._b,
                              shape = shape)


class GeneratorTriangular(IGeneratorRandom):

    __slots__ = ("_left","_mode","_right","_rng")

    def __init__(self,
                 left: float = -1.0,
                 mode: float = 0.0,
                 right: float = 1.0,
                 seed: int | None = None) -> None:

        self._left = left
        self._mode = mode
        self._right = right

        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int,...]) -> np.ndarray:
        return self._rng.triangular(left = self._left,
                                    mode = self._mode,
                                    right = self._right,
                                    shape = shape)