"""
module docstring

author: Scott Warchal
"""

import random
from collections import namedtuple
from functools import reduce
import operator
from typing import Tuple, List, Union, Generator

import numpy as np
import plate_generator
from plate_generator import core
from .core import Plate


__all__ = ["generator", "generator_combined"]


func_tuple = namedtuple("FuncTuple", ["func", "int"])

ALL_PLATE_FUNCTIONS = [
    func_tuple(core.normal_plate,         0),
    func_tuple(core.uniform_plate,        0),
    func_tuple(core.lognormal_plate,      0),
    func_tuple(core.edge_plate,           1),
    func_tuple(core.row_plate,            2),
    func_tuple(core.column_plate,         3),
    func_tuple(core.single_checker_plate, 4),
    func_tuple(core.quad_checker_plate,   5),
    func_tuple(core.h_grad_plate,         6),
    func_tuple(core.v_grad_plate,         7)
    # TODO:
    # func_tuple(core.bleedthrough_plate, 8),
    # func_tuple(core.snake_plate,        9)
]


def generator(n: int, size=1536, effects="all", **kwargs) -> Generator:
    """
    plate generator to create random plates with various
    artefacts on the fly

    Parameters:
    -----------
    n: number of samples
    size: plate size (number of wells)
        if "either" or "both" then will randomly return a
        384 or 1536 well plate
    effects: str or list
        which plates to generate

    Returns:
    --------
    A generator which returns a named tuple:
        output.plate : Plate class
        output.int   : index of label
    """
    sizes = {384, 1536}
    for i in range(n):
        # NOTE: if needed this can be sped up by creating separate loops
        # for either plate option (known or random), rather than checking
        # the value of size at each iteration, but it will make the code
        # more complicated.
        plate_size = size if size in sizes else random.sample(sizes, 1)[0]
        yield create_output_tuple(plate_size, effects, **kwargs)


def generator_combined(n: int,
                       size: int = 1536,
                       effects: Union[str, List] = "all",
                       op: str = "+",
                       max_combinations: int = 2,
                       zeta: float = 2.0,**kwargs) -> Generator:
    """
    Generate plates with combined effects, for example a plate with
    an edge_effect + column_effect.
    This is useful for a multi-label classifier, as plate effects are
    not usually mutually independent.

    Parameters:
    -----------
    n: number of samples
    size: plate size (number of wells)
        if "either" or "both" then will randomly return a
        384 or 1536 well plate
    effects: str or list
        which plates to generate
    op: str, options: ['+', '*']
        operator, how to combine plates, options are either add '+' or
        multiply '*' plates together.
    max_combinations: int
        maximum number of effects which can be combined.
    zeta: float
        parameter passed to numpy.random.zipf() to construct the probabilitiy
        distribution from which the number of plate combinations is drawn from.

    Returns:
    --------
    A generator which returns a named tuple:
        output.plate : Plate class
        output.label : string, name of plate effect
    """
    sizes = {384, 1536}
    if isinstance(effects, list):
        plate_functions = [func_tuple(f, i) for i, f in enumerate(effects)]
    elif effects == "all":
        plate_functions = ALL_PLATE_FUNCTIONS
    # set operator based on function argument
    if op == "+":
        op_func = operator.add
    elif op == "*":
        op_func = operator.mul
    else:
        raise ValueError("invalid operator, options: ['+', '*']")
    plate_tuple = namedtuple("output", ["plate", "int"])
    prob_dist = np.random.zipf(zeta, 1000)
    # trim probability distribution function (prob_dist) to a maximum of
    # `max_combinations`
    within_range = (prob_dist <= max_combinations)
    prob_dist = prob_dist[within_range]
    count = 0
    while count < n:
        n_combinations = random.sample(prob_dist.tolist(), 1)[0]
        plate_size = size if size in sizes else random.sample(sizes, 1)[0]
        if n_combinations == 1:
            # then just get a single plate from the generator
            # but put the labels and ints inside a list so that all the
            # tuples returned by the generator are consistent
            yield create_output_tuple(plate_size, effects, listify=True, **kwargs)
            count += 1
        else:
            random_effect_list = random.sample(
                plate_functions, n_combinations
            )
            # now have a list of namedtuples
            funcs  = [i.func for i in random_effect_list]
            plates = [f(plate=plate_size, **kwargs) for f in funcs]
            # add or multiple plates together
            effect_plate  = reduce(lambda x, y: op_func(x, y), plates)
            # if random is an effect to be combined then skip this current
            # iteration and don't increment the counter.
            # otherwise you end up with a multi-label class that includes
            # "random" as one of the classes, which is just the non-random
            # classes with more noise, which is not useful as a label
            print(effect_plate.name)
            if "random" not in effect_plate.name:
                effect_ints   = [i.int for i in random_effect_list]
                yield plate_tuple(effect_plate, effect_ints)
                count += 1


def create_output_tuple(size, effects, listify=False, **kwargs):
    """
    Creates an output tuple containing a plate, labels,
    and integer labels.

    Parameters:
    -----------
    size: int
        plate size (number of wells)
    listify: bool
        If True then  labels and itegers will be wrapped in a list.
        This is useful for generator_combined so all returned tuples
        are consistent regardless if they have been combined or not.
    **kwargs:
        additional arguments to core plate generating functions
    Returns:
    --------
    A generator which returns a named tuple:
        output.plate : Plate class
        output.int   : index of label
    """
    if effects == "all":
        plate_functions = ALL_PLATE_FUNCTIONS
    elif isinstance(effects, list):
        plate_functions = [func_tuple(f, i) for i, f in enumerate(effects)]
    else:
        raise ValueError
    plate_tuple = namedtuple("output", ["plate", "int"])
    random_effect = random.sample(plate_functions, 1)[0]
    effect_func, effect_int = random_effect
    effect_plate = effect_func(plate=size, **kwargs)
    if listify:
        effect_int = [effect_int]
    return plate_tuple(effect_plate,  effect_int)

