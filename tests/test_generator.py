"""
module docstring
"""
from collections import namedtuple, Counter

import plate_generator
import pytest


def test_generator():
    my_generator = plate_generator.generator(n=10)
    for plate, name, int_label in my_generator:
        assert isinstance(plate, plate_generator.Plate)


def test_generator_size():
    my_generator_384 = plate_generator.generator(n=10, size=384)
    for plate, name, int_label in my_generator_384:
        assert plate.size == 384
        assert plate.shape == (16, 24)
    ##
    my_generator_1536 = plate_generator.generator(n=10, size=1536)
    for plate, name, int_label in my_generator_1536:
        assert plate.size == 1536
        assert plate.shape == (32, 48)


def test_generator_n():
    """check it creates the correct number of plates"""
    count = 0
    N = 1000
    my_generator = plate_generator.generator(n=N)
    for i in my_generator:
        count += 1
    assert count == N


def test_generator_random_size():
    count_384 = 0
    count_1536 = 0
    n = 100
    my_generator = plate_generator.generator(n=n, size="either")
    for plate, name, _ in my_generator:
        plate_size = plate.size
        if plate_size == 384:
            count_384 += 1
        if plate_size == 1536:
            count_1536 += 1
    assert count_384 > 0
    assert count_1536 > 0
    assert count_384 + count_1536 == n


def test_generator_select_size():
    for output in plate_generator.generator(n=1000, size=384):
        assert output.plate.size == 384
    for output in plate_generator.generator(n=1000, size=1536):
        assert output.plate.size == 1536


def test_generator_combined():
    my_generator = plate_generator.generator_combined(n=1000)
    for output in my_generator:
        assert isinstance(output.plate, plate_generator.core.Plate)
        assert isinstance(output.label, list)
        assert isinstance(output.int, list)
    my_generator = plate_generator.generator_combined(n=1000)
    # check that at least some of the plates have been combined
    combined = 0
    for output in my_generator:
        assert len(output.label) == len(output.int)
        if len(output.label) > 1:
            combined += 1
    assert combined > 0


def test_generator_combined_correct_n():
    count = 0
    n = 1000
    for i in plate_generator.generator_combined(n=n):
        count += 1
    assert count == n


def test_generator_combined_max_combinations():
    max_combinations = 3
    my_generator = plate_generator.generator_combined(
        n=1000,
        max_combinations=max_combinations
    )
    for output in my_generator:
        assert len(output.label) <= max_combinations
    ###########
    max_combinations = 1
    my_generator = plate_generator.generator_combined(
        n=1000,
        max_combinations=max_combinations
    )
    for output in my_generator:
        assert len(output.label) <= max_combinations
    #####
    max_combinations = 5
    my_generator = plate_generator.generator_combined(
        n=1000,
        max_combinations=max_combinations
    )
    for output in my_generator:
        assert len(output.label) <= max_combinations


def test_generator_combined_size():
    size = 384
    my_generator = plate_generator.generator_combined(1000, size=384)
    for output in my_generator:
        assert output.plate.size == 384


def test_generator_combined_raises_invalid_op():
    with pytest.raises(ValueError):
        # NOTE: doesn't error on creating the variable, only when
        # actually iterating through the generator
        my_generator = plate_generator.generator_combined(100, op="//")
        for output in my_generator:
            pass


def test_generator_combined_random_size():
    my_generator = plate_generator.generator_combined(1000, size="either")
    sizes = []
    for plate in my_generator:
        sizes.append(plate.plate.size)
    assert len(Counter(sizes)) == 2


