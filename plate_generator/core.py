"""
Generate synthetic data for plate maps, including common plate effects.

author: Scott Warchal
date  : 2019-01-31
"""

import operator
import random
from typing import Tuple, Optional, Union
import warnings

import numpy as np
import scipy.ndimage


class Plate:
    """plate class"""
    def __init__(self,
                 data: np.ndarray,
                 name: Union[List, str],
                 noise: Optional[np.array] = None,
                 effect: Optional[np.array] = None,
                 size: int = 385):
        if size not in [384, 1536]:
            raise ValueError("invalid size. options: [384, 1536]")
        self.data = data
        self.noise = noise
        if effect is None:
            effect = np.zeros_like(data)
        self.effect = effect
        self.size = size
        assert self.data.size == self.size
        # shape is (height, width) following numpy convention
        self.shape = (16, 24) if size == 384 else (32, 48)
        # name has to be in a list as plates may be combined
        if isinstance(name, str):
            self.name = [name]
        elif isinstance(name, list):
            self.name = name
        else:
            raise ValueError("name has to be a string or a list")

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __str__(self):
        return self.data

    def apply_op(self, other, op):
        if isinstance(other, Plate):
            new_data = op(self.data, other.data)
            new_noise = op(self.noise, other.noise)
            new_effect = op(self.effect, other.effect)
            new_name = self.name
            new_name.extend(other.name)
        elif isinstance(other, (np.ndarray, int, float)):
            warnings.warn("data is no longer a combination of noise & effect")
            new_data = op(self.data, other)
            new_noise = self.noise
            new_effect = self.effect
            new_name = self.name
        else:
            raise TypeError
        return Plate(data=new_data, noise=new_noise, effect=new_effect,
                     name=new_name, size=self.size)

    def __add__(self, other):
        return self.apply_op(other, operator.add)

    def __sub__(self, other):
        return self.apply_op(other, operator.sub)

    def __mul__(self, other):
        return self.apply_op(other, operator.mul)

    def __truediv__(self, other):
        return self.apply_op(other, operator.truediv)

    def __floordiv__(self, other):
        return self.apply_op(other, operator.floordiv)

    # this makes the operators commutative
    __rmul__ = __mul__
    __rtruediv__ = __truediv__
    __rfloordiv__ = __floordiv__
    __radd__ = __add__
    __rsub__ = __sub__
    # this is needed for commutative numpy operations
    __array_ufunc__ = None

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
        return Plate(data=norm_data, name=self.name, size=self.size)

    def _normalise(self):
        """in-place normalisation"""
        self.data = (self.data - self.mean()) / self.std()
        return self


def get_sigma(sigma: Optional[float], low=0.1, high=5) -> float:
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
    return Plate(data=x.data, name=x.name, size=x.size)


def _random_edge(num_edges: int,
                 shape: Tuple[int, int],
                 sigma: float) -> np.array:
    """
    internal function for edge_plate2()
    """
    assert num_edges in [2, 3, 4], "invalid number of edges"
    edge_plate = np.full(shape=shape, fill_value=sigma*10)
    edge_plate[1:-1, 1:-1] = 0
    possible_edges = [np.index_exp[0,  :],  # top edge
                      np.index_exp[-1, :],  # bottom edge
                      np.index_exp[:,  0],  # left edge
                      np.index_exp[:, -1]]  # right edge
    # if num_edges == 4 then just return the edge_plate
    # otherwise:
    if num_edges in [2, 3]:
        # randomly set one of the edges to zeros
        for edge in random.sample(possible_edges, 4 - num_edges):
            edge_plate[edge] = 0
    return edge_plate


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
    noise = np.random.normal(loc=0, scale=sigma, size=input_plate.shape)
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
    return Plate(data=data, noise=data, name="random", size=plate)


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
    return Plate(data=data, noise=data, name="random", size=plate)


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
    return Plate(data=data, noise=data, name="random", size=plate)


def edge_plate(plate: int = 384,
               sigma: Optional[float] = None,
               invert: Optional[bool] = False) -> Plate:
    """Edge plate, randomly chosen from the 2 edge plate models."""
    f = random.sample([edge_plate1, edge_plate2], 1)[0]
    return f(plate, sigma, invert)


def edge_plate1(plate: int = 384,
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
    edge_plate[1:-1, -2] = 2  # right inner
    if invert:
        edge_plate = 1 - edge_plate
    data = edge_plate + noise_plate
    return Plate(data=data, noise=noise_plate,
                 effect=edge_plate, name="edge", size=plate)


def edge_plate2(plate: int = 384,
                sigma: Optional[float] = None,
                spread: Optional[float] = None,
                invert: Optional[float] = None,
                randomise_edges: bool = False) -> Plate:
    """
    Create a plate with an edge effect.
    Use a diffusion model to influence edge wells with values beyond
    the edge of the plate.
    Parameters:
    -----------
    plate: int
        number of wells
    sigma: float or None
        standard deviation of the noise.
        If `None` then will be randomly sampled.
    spread: float
        How far a well's signal diffuses into adjacent wells. This is
        standard deviation of the Gaussian kernel used in the blur operation.
    invert: bool or None
        Whether to invert the plate values or not.
        If `None` then will be done randomly.
    randomise_edges: bool
        if True then not all edges will be subjected to an edge effect, instead
        between 2 and 4 edges will be randomly subjected to an edge effect.
        Note that a single effected edge will be undistinguishable from a
        gradient effect, and so is not included as a possibility.
    Returns:
    ---------
    Plate
    """
    sigma = get_sigma(sigma)
    invert = get_invert(invert)
    shape = size2shape(plate)
    if spread is None:
        # randomly sample spread value
        spread = np.random.uniform(low=1, high=8, size=1)[0]
    # add an outer well on all edges
    diffuse_shape = [i+2 for i in shape]
    if randomise_edges:
        # randomise the number of edges to have an effect
        n_edges = random.sample([2, 3, 4], 1)[0]
        edge_plate = _random_edge(
            n_edges, shape=diffuse_shape, sigma=sigma
        )
    else:
        # all edges will have an edge effect
        edge_plate = np.full(shape=diffuse_shape, fill_value=sigma*10)
        # set inner wells to zero
        edge_plate[1:-1, 1:-1] = 0
    if invert:
        edge_plate = 1 - edge_plate
    effect_plate = scipy.ndimage.gaussian_filter(edge_plate, spread)
    # remove edges to return back to the intended plate size
    effect_plate = effect_plate[1:-1, 1:-1]
    noise_plate = np.random.normal(loc=0, scale=sigma, size=shape)
    data = effect_plate + noise_plate
    return Plate(data=data, noise=noise_plate,
                 effect=effect_plate, name="edge", size=plate)


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
    return Plate(data=data, noise=noise_plate,
                 effect=effect_plate, name="row", size=plate)


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
    return Plate(data=data, noise=noise_plate,
                 effect=effect_plate, name="column", size=plate)


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
    return Plate(data=data, noise=noise_plate,
                 effect=effect_plate, name="single_checker", size=plate)


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
    effect_plate = (
        np.random.normal(loc=0, scale=sigma, size=half_shape)
            .repeat(2, axis=0)
            .repeat(2, axis=1)
    )
    noise_plate = np.random.normal(loc=0, scale=sigma, size=shape)
    data = noise_plate + effect_plate
    return Plate(data=data, noise=noise_plate,
                 effect=effect_plate, name="quad_checker", size=plate)


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
    return Plate(data=data, noise=noise_plate,
                 effect=effect_plate, name="horizontal_gradient", size=plate)


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
    return Plate(data=data, noise=noise_plate,
                 effect=effect_plate, name="vertical_gradient", size=plate)


def bleed_through_plate(plate: int = 384,
                        prop: float = 0.05,
                        high_vals: float = 35.0,
                        spread: float = 0.75,
                        sigma: Optional[float] = None) -> Plate:
    """
    Create a plate with bleed-through artefacts where a strong signal in
    a well influences neighbouring wells.
    Parameters:
    -----------
    plate: int
        number of wells
    prop: float
        proportion of the plate which should be extremely high values
    high_vals: float
        The value for the strong signal, the background is centered on zero
        while wells with a "high value" will be given this value.
    spread: float
        How far a well's signal diffuses into adjacent wells. This is
        standard deviation of the Gaussian kernel used in the blur operation.
    sigma: float or None
        standard deviation of the noise.
        If `None` then will be randomly sampled.
    Returns:
    --------
    Plate
    """
    sigma = get_sigma(sigma)
    shape = size2shape(plate)
    num_high_wells = int(plate * prop)
    effect_plate = np.zeros(plate)
    high_well_idx = random.sample(range(effect_plate.size), num_high_wells)
    effect_plate[high_well_idx] = high_vals
    effect_plate = effect_plate.reshape(shape)
    effect_plate = scipy.ndimage.gaussian_filter(effect_plate, spread)
    noise_plate = np.random.normal(loc=0, scale=sigma, size=shape)
    data = noise_plate + effect_plate
    # FIXME: want the noise plate to also contain the high values, and the
    #        effect to only contain the bleed-through effect
    return Plate(data=data, noise=noise_plate,
                 effect=effect_plate, name="bleedthrough", size=plate)


def snake_plate(plate: int = 384,
                sigma: Optional[float] = None,
                row_wise: bool = True,
                max_lim: float = 50.0,
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
    row_wise: bool
        TODO
    max_lim: float
        TODO
    direction: bool (optional)
        direction of sequential order
    Returns:
    --------
    Plate
    """
    raise NotImplementedError(
        "not working very well, pretty much just a gradient plate"
    )
    sigma = get_sigma(sigma)
    shape = size2shape(plate)
    effect_plate = np.linspace(0, max_lim, plate)
    if row_wise:
        effect_plate = effect_plate.reshape(shape)
        effect_plate[1::2, :] = effect_plate[1::2, ::-1]
    else:  # column-wise
        effect_plate = effect_plate.reshape(shape[::-1]).T
        effect_plate[:, 1::2] = effect_plate[::-1, 1::2]
    noise_plate = np.random.normal(loc=0, scale=sigma, size=shape)
    data = noise_plate + effect_plate
    return Plate(data=data, name="snake", size=plate)
