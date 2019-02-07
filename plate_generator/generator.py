"""
module docstring

author: Scott Warchal
date  : 2019-02-07
"""

import random

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
    # plate_generator.bleedthrough_plate,
    # plate_genertor.snake_plate
]


def generator(n :int, size :int = 1536, effects="all", **kwargs):
    """
    plate generator to create random plates with various
    artefacts on the fly

    Parameters:
    -----------
    n: number of samples
    size: plate size (number of wells)
    effects: str or list
        which plates to generate

    Returns:
    --------
    generator
    """
    if effects == "all":
        n_types_of_effects = len(ALL_PLATE_FUNCTIONS)
        plate_functions = ALL_PLATE_FUNCTIONS
    else:
        # create new plate_functions
        raise NotImplementedError
    for i in range(n):
        random_effect = random.sample(range(n_types_of_effects), 1)[0]
        effect_func, effect_name = plate_functions[random_effect]
        effect_plate = effect_func(plate=size, **kwargs)
        yield effect_plate, effect_name



