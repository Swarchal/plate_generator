"""
module docstring

author: Scott Warchal
"""

import random
from collections import namedtuple
from functools import reduce
import operator

import numpy as np
import plate_generator

func_tuple = namedtuple("FuncTuple", ["func", "label", "int"])

ALL_PLATE_FUNCTIONS = [
    func_tuple(plate_generator.normal_plate,         "random",              0),
    func_tuple(plate_generator.uniform_plate,        "random",              0),
    func_tuple(plate_generator.lognormal_plate,      "random",              0),
    func_tuple(plate_generator.edge_plate,           "edge",                1),
    func_tuple(plate_generator.row_plate,            "row",                 2),
    func_tuple(plate_generator.column_plate,         "column",              3),
    func_tuple(plate_generator.single_checker_plate, "single_checker",      4),
    func_tuple(plate_generator.quad_checker_plate,   "quad checker",        5),
    func_tuple(plate_generator.h_grad_plate,         "horizontal_gradient", 6),
    func_tuple(plate_generator.v_grad_plate,         "vertical_gradient",   7)
    # TODO:
    # func_tuple(plate_generator.bleedthrough_plate, "bleedthrough",        8),
    # func_tuple(plate_generator.snake_plate,         "snake",              9)
]


def generator(n :int, size=1536, effects="all", **kwargs):
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
        output.label : string, name of plate effect
    """
    if effects != "all":
        raise NotImplementedError
    plate_tuple = namedtuple("output", ["plate", "label", "int"])
    if isinstance(size, int):
        # use size directly, much quicker than checking the type
        # of `size` each iteration
        for i in range(n):
            random_effect = random.sample(ALL_PLATE_FUNCTIONS, 1)[0]
            effect_func, effect_name, effect_int = random_effect
            effect_plate = effect_func(plate=size, **kwargs)
            yield plate_tuple(effect_plate, effect_name, effect_int)
    elif size in ["either", "both"] or isinstance(size, list):
        for i in range(n):
            random_effect = random.sample(ALL_PLATE_FUNCTIONS, 1)[0]
            effect_func, effect_name, effect_int = random_effect
            # randomly sample size at each iteration
            size = random.sample([384, 1536], 1)[0]
            effect_plate = effect_func(plate=size, **kwargs)
            yield plate_tuple(effect_plate, effect_name, effect_int)
    else:
        raise ValueError


def generator_combined(n, size=1536, effects="all", op="+",
                       max_combinations=2, zeta=2.0,**kwargs):
    """
    Generate plates with combined effects, for example a plate with
    an edge_effect + column_effect.
    This is useful for a multi-label classifier.

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
    if effects != "all":
        raise NotImplementedError
    # set operator based on function argument
    if op == "+":
        op = operator.add
    elif op == "*":
        op = operator.mul
    else:
        raise ValueError("invalid operator, options: ['+', '*']")
    plate_tuple = namedtuple("output", ["plate", "label", "int"])
    prob_dist = np.random.zipf(zeta, 1000)
    # trim probability distribution function (prob_dist) to a maximum of
    # `max_combinations`
    within_range = (prob_dist <= max_combinations)
    prob_dist = prob_dist[within_range]
    count = 0
    if isinstance(size, int):
        while count < n:
            n_combinations = random.sample(prob_dist.tolist(), 1)[0]
            if n_combinations == 1:
                # then just get a single plate from the generator
                random_effect = random.sample(ALL_PLATE_FUNCTIONS, 1)[0]
                effect_func, effect_name, effect_int = random_effect
                effect_plate = effect_func(plate=size, **kwargs)
                yield plate_tuple(effect_plate, [effect_name], [effect_int])
                count += 1
            else:
                # FIXME: this is too clever: I won't understand it in 2 weeks
                #
                # sample multiple plates from the generator
                # and add them together
                random_effect_list = random.sample(
                        ALL_PLATE_FUNCTIONS, n_combinations
                )
                # now have a list of namedtuples
                funcs  = [i.func for i in random_effect_list]
                plates = [f(plate=size, **kwargs) for f in funcs]
                # add or multiple plates together
                effect_plate  = reduce(lambda x, y: op(x, y), plates)
                effect_labels = [i.label for i in random_effect_list]
                # if random is one of the effects to be combined
                # then skip this current iteration
                if "random" not in effect_labels:
                    effect_ints   = [i.int for i in random_effect_list]
                    yield plate_tuple(effect_plate, effect_labels, effect_ints)
                    count += 1
    elif size in ["either", "both"] or isinstance(size, list):
        # TODO: make this
        raise NotImplementedError
    else:
        raise TypeError

