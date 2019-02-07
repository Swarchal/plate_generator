"""
module docstring

author: Scott Warchal
date  : 2019-02-07
"""

import random
from collections import namedtuple

import plate_generator


ALL_PLATE_FUNCTIONS = [
    (plate_generator.normal_plate, "random"),
    (plate_generator.uniform_plate, "random"),
    (plate_generator.lognormal_plate, "random"),
    (plate_generator.edge_plate, "edge"),
    (plate_generator.row_plate, "row"),
    (plate_generator.column_plate, "column"),
    (plate_generator.single_checker_plate, "single_checker"),
    (plate_generator.quad_checker_plate, "quad checker"),
    (plate_generator.h_grad_plate, "horizontal_gradient"),
    (plate_generator.v_grad_plate, "vertical_gradient")
    # TODO:
    # (plate_generator.bleedthrough_plate, "bleedthrough"),
    # (plate_genertor.snake_plate, "snake")
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
    if effects == "all":
        n_types_of_effects = len(ALL_PLATE_FUNCTIONS)
        plate_functions = ALL_PLATE_FUNCTIONS
    else:
        # create new plate_functions
        raise NotImplementedError
    plate_tuple = namedtuple("output", ["plate", "label"])
    if isinstance(size, int):
        # use size directly, much quicker than checking the type
        # of `size` each iteration
        for i in range(n):
            random_effect = random.sample(range(n_types_of_effects), 1)[0]
            effect_func, effect_name = plate_functions[random_effect]
            effect_plate = effect_func(plate=size, **kwargs)
            yield plate_tuple(effect_plate, effect_name)
    elif size in ["either", "both"]:
        for i in range(n):
            # randomly sample size at each itertion
            size = random.sample([384, 1536], 1)[0]
            random_effect = random.sample(range(n_types_of_effects), 1)[0]
            effect_func, effect_name = plate_functions[random_effect]
            effect_plate = effect_func(plate=size, **kwargs)
            yield plate_tuple(effect_plate, effect_name)
    else:
        raise ValueError


