"""
Generate synthetic data for plate maps, including common plate effects.

author: Scott Warchal
date  : 2019-01-31
"""

import random
from typing import List, Tuple, Optional

import numpy as np


class Plate:
    """plate class"""

    def __init__(self, data: np.ndarray, size: int = 385):
        if size not in [384, 1536]:
            raise ValueError("invalid size. options: [384, 1536]")
        self.data = data
        self.size = size
        # shape is (height, width) following numpy convention
        self.shape = (16, 24) if size == 384 else (32, 48)

    @property
    def ndim(self):
        return self.data.ndim

    def __str__(self):
        return self.data

    def __add__(self, other):
        if isinstance(other, Plate):
            new_data = self.data + other.data
        elif isinstance(other, (np.ndarray, int, float)):
            new_data = self.data + other
        else:
            raise TypeError
        return Plate(data=new_data, size=self.size)

    def __sub__(self, other):
        if isinstance(other, Plate):
            new_data = self.data - other.data
        elif isinstance(other, (np.ndarray, int, float)):
            new_data = self.data - other
        else:
            raise TypeError
        return Plate(data=new_data, size=self.size)

    def __mul__(self, other):
        if isinstance(other, Plate):
            new_data = self.data * other.data
        elif isinstance(other, (np.ndarray, int, float)):
            new_data = self.data * other
        else:
            raise TypeError
        return Plate(data=new_data, size=self.size)

    def __truediv__(self, other):
        if isinstance(other, Plate):
            new_data = self.data / other.data
        elif isinstance(other, (np.ndarray, int, float)):
            new_data = self.data / other
        else:
            raise TypeError
        return Plate(data=new_data, size=self.size)

    # FIXME: these are not working as expected
    # this makes the operators commutative
    __rmul__ = __mul__
    __rdiv__ = __truediv__
    __radd__ = __add__
    __rsub__ = __sub__

    def mean(self):
        return self.data.mean()

    def std(self):
        return self.data.std()

    def normalise(self):
        """
        return a new Plate with Plate.data normalised to a mean of 0
        and standard deviation of 1
        """
        norm_data = (self.data - self.mean()) / self.std()
        return Plate(data=norm_data, size=self.size)

    def _normalise(self):
        """in-place normalisation"""
        self.data = (self.data - self.mean()) / self.std()
        return self


def get_sigma(sigma: float, low=0.1, high=5) -> float:
    """
    Automatically generate a sigma value if missing (None).
    If sigma is not missing, then simply return the input value.
    The generated value is randomly sampled from a uniform distribution
    between `low` and `high`.
    Parameters:
    -----------
        sigma: float or None
            sigma value, if supplied then this will be the return value
        low: numeric
            lower-bound for the sigma value if generated
        high: numeric
            upper-bound for the sigma value
    Returns:
    ---------
    float
    """
    if sigma is None:
        sigma = np.random.uniform(low, high)
    return float(sigma)


def get_invert(invert: Optional[bool]) -> bool:
    """
    Automatically generate an invert value if missing (None).
    If invert is not missing, then simply return the input value
    """
    if invert is None:
        invert = random.sample([True, False], 1)[0]
    return invert


def size2shape(size: int) -> Tuple[int, int]:
    """convert a plate size (number of wells) to shape (y, x)"""
    assert size in [384, 1536]
    return (16, 24) if size == 384 else (32, 48)


def normalise(x: Plate) -> Plate:
    """normalise (standard scale) a numpy array"""
    x.data = (x.data - x.data.mean()) / x.data.std()
    return Plate(data=x.data, size=x.size)


def add_noise(input_plate: Plate,
              sigma: Optional[float] = None, **kwargs) -> Plate:
    """
    Add random (normal) noise to a plate.
    Parameters:
    -----------
    input_plate: Plate
    sigma: float or None
        standard deviation for the noise.
        If `None` then will be sampled randomly.
    **kwargs:
        additional keyword arguments to `get_sigma`
    Returns:
    --------
    Plate
    """
    # generate noise in the same shape as input_plate
    sigma = get_sigma(sigma, **kwargs)
    noise = np.random.normal(loc=0, scale=sigma, shape=input_plate.shape)
    # add noise to existing data in input_plate
    input_plate.data = input_plate.data + noise
    return input_plate


def normal_plate(plate: int = 384,
                 sigma: Optional[float] = None) -> Plate:
    """
    Random plate from a normal distribution.
    Parameters:
    -----------
    plate: int
        number of wells
    sigma: float or None
        standard deviation for the noise.
        If `None` then will be sampled randomly.
    Returns:
    --------
    Plate
    """
    sigma = get_sigma(sigma)
    shape = size2shape(plate)
    data = np.random.normal(loc=0, scale=sigma, size=shape)
    return Plate(data=data, size=plate)


def uniform_plate(plate: int = 384, low=0.1, high=3) -> Plate:
    """
    Random plate from a uniform distribution.
    Parameters:
    -----------
    plate: int
        number of wells
    low: float
        lower limit of uniform distribution
    high: float
        upper limit of uniform distribution
    Returns:
    --------
    Plate
    """
    shape = size2shape(plate)
    data = np.random.uniform(low, high, size=shape)
    return Plate(data=data, size=plate)


def lognormal_plate(plate: int = 384,
                    sigma: Optional[float] = None) -> Plate:
    """
    Random plate from a log distribution.
    Parameters:
    -----------
    plate: int
        number of wells
    sigma: float or None
        standard deviation for the noise.
        If `None` then will be sampled randomly.
    Returns:
    --------
    Plate
    """
    sigma = get_sigma(sigma)
    shape = size2shape(plate)
    data = np.random.lognormal(mean=0, sigma=sigma, size=shape)
    return Plate(data=data, size=plate)


def edge_plate(plate: int = 384,
               sigma: Optional[float] = None,
               invert: Optional[bool] = False) -> Plate:
    """
    Create a plate with an edge effect.
    Outer two rows/columns should show an effect.
    Parameters:
    -----------
    plate: int
        number of wells
    sigma: float or None
        standard deviation of the noise.
        If `None` then will be randomly sampled.
    invert: bool or None
        Whether to invert the plate values or not.
        If `None` then will be done randomly.
    """
    sigma = get_sigma(sigma)
    invert = get_invert(invert)
    shape = size2shape(plate)
    edge_plate = np.zeros(shape=shape)
    noise_plate = np.random.normal(loc=0, scale=sigma, size=shape)
    # outer
    edge_plate[0, :] = 3  # top edge
    edge_plate[-1, :] = 3   # bottom edge
    edge_plate[:, 0] = 3  # left edge
    edge_plate[1:-1, -1] = 3  # right edge
    # inner
    edge_plate[1, 1:-1] = 2  # top inner
    edge_plate[-2, 1:-1] = 2  # bottom inner
    edge_plate[1:-1, 1] = 2  # left inner
    edge_plate[1:-1, -2] = 2
    if invert:
        raise NotImplementedError
    #    noise_plate = 1 / noise_plate
    data = edge_plate + noise_plate
    return Plate(data=data, size=plate)


def row_plate(plate: int = 384,
              sigma: Optional[float] = None) -> Plate:
    """
    Create a plate with row effects.
    With alternating rows of higher/lower values.
    Parameters:
    -----------
    plate: int
        number of wells
    sigma: float or None
        standard deviation of the noise.
        If `None` then will be randomly sampled.
    Returns:
    --------
    Plate
    """
    sigma = get_sigma(sigma)
    shape = size2shape(plate)
    noise_plate = np.random.normal(loc=0, scale=sigma, size=shape)
    effect_plate = np.ones(shape) * 1.5
    effect_plate[::2, :] = 0
    data = effect_plate + noise_plate
    return Plate(data=data, size=plate)


def column_plate(plate: int = 384,
                 sigma: Optional[float] = None) -> Plate:
    """
    Create a plate with column effects.
    With alternating columns of higher/lower values.
    Parameters:
    -----------
    plate: int
        number of wells
    sigma: float or None
        standard deviation of the noise.
        If `None` then will be randomly sampled.
    Returns:
    --------
    Plate
    """
    sigma = get_sigma(sigma)
    shape = size2shape(plate)
    noise_plate = np.random.normal(loc=0, scale=sigma, size=shape)
    effect_plate = np.ones(shape) * 1.5
    effect_plate[:, ::2] = 0
    data = effect_plate + noise_plate
    return Plate(data=data, size=plate)


def single_checker_plate(plate: int = 384,
                         sigma: Optional[float] = None) -> Plate:
    """
    Create a plate with single-well checker effects.
    Parameters:
    -----------
    plate: int
        number of wells
    sigma: float or None
        standard deviation of the noise.
        If `None` then will be randomly sampled.
    Returns:
    --------
    Plate
    """
    sigma = get_sigma(sigma)
    shape = size2shape(plate)
    effect_plate = np.ones(shape) * 5
    effect_plate[::2, ::2] = 0
    noise_plate = np.random.normal(loc=0, scale=sigma, size=shape)
    data = noise_plate + effect_plate
    return Plate(data=data, size=plate)


def quad_checker_plate(plate: int = 384,
                       sigma: Optional[float] = None) -> Plate:
    """
    Create a plate with 4-well checker effects
    Parameters:
    -----------
    plate: int
        number of wells
    sigma: float or None
        standard deviation of the noise.
        If `None` then will be randomly sampled.
    Returns:
    --------
    Plate
    """
    sigma = get_sigma(sigma)
    shape = size2shape(plate)
    half_shape = [i//2 for i in shape]
    # expand plate with half-sized up dimensions two-fold in both axes
    # this creates a plate with repeating 4-element blocks
    checker_plate = (
        np.random.normal(loc=0, scale=sigma, size=half_shape)
            .repeat(2, axis=0)
            .repeat(2, axis=1)
    )
    noise_plate = np.random.normal(loc=0, scale=sigma, size=shape)
    data = noise_plate + checker_plate
    return Plate(data=data, size=plate)


def h_grad_plate(plate: int = 384,
                 sigma: Optional[float] = None,
                 flip: Optional[bool] = None) -> Plate:
    """
    Create a plate with a horizontal gradient
    Parameters:
    -----------
    plate: int
        number of wells
    sigma: float or None
        standard deviation of the noise.
        If `None` then will be randomly sampled.
    Flip: bool or None
        whether to horizontally flip the array.
        If `None` then will be flipped randomly.
    Returns:
    --------
    Plate
    """
    sigma = get_sigma(sigma)
    shape = size2shape(plate)
    if flip is None:
        flip = random.sample([True, False], 1)[0]
    effect_plate = (
        np.linspace(-sigma * 1.5, sigma * 1.5, plate)
            .reshape(*shape[::-1])
            .transpose()
    )
    if flip:
        effect_plate = np.flip(effect_plate, axis=1)
    noise_plate = np.random.normal(loc=0, scale=sigma, size=shape)
    data = noise_plate + effect_plate
    return Plate(data=data, size=plate)


def v_grad_plate(plate: int = 384,
                 sigma: Optional[float] = None,
                 flip: Optional[bool] = None) -> Plate:
    """
    Create a plate with a vertical gradient
    Parameters:
    -----------
    plate: int
        number of wells
    sigma: float or None
        standard deviation of the noise.
        If `None` then will be randomly sampled.
    flip: bool or None
        whether to vertically flip the array
        If `None` then will be flipped randomly
    Returns:
    --------
    Plate
    """
    sigma = get_sigma(sigma)
    shape = size2shape(plate)
    if flip is None:
        flip = random.sample([True, False], 1)[0]
    effect_plate = (
        np.linspace(-sigma * 1.5, sigma * 1.5, plate)
            .reshape(*shape)
    )
    if flip:
        effect_plate = np.flip(effect_plate, axis=0)
    noise_plate = np.random.normal(loc=0, scale=sigma, size=shape)
    data = noise_plate + effect_plate
    return Plate(data=data, size=plate)


def bleed_through_plate(plate: int = 384,
                        sigma: Optional[float] = None) -> Plate:
    """
    Create a plate with bleed-through artefacts where a strong signal in
    a well influences neighbouring wells.
    Parameters:
    -----------
    plate: int
        number of wells
    sigma: float or None
        standard deviation of the noise.
        If `None` then will be randomly sampled.
    Returns:
    --------
    Plate
    """
    raise NotImplementedError()


def snake_plate(plate: int = 384,
                sigma: Optional[float] = None,
                direction: Optional[str] = None) -> Plate:
    """
    Create a plate with a snake-like pattern caused by sequentially
    dispensing into adjacent wells.
    Parameters:
    -----------
    plate: int
        number of wells
    sigma: float or None
        standard deviation of the noise.
        If `None` then will be randomly sampled.
    Returns:
    --------
    Plate
    """
    raise NotImplementedError()

