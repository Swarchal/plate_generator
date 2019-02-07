import os
import plate_generator

import numpy as np


CURRENT_PATH = os.path.dirname(__file__)
SHAPE_384 = (16, 24)
SHAPE_1536 = (32, 48)
EPS = 1e-6


def test_normal_plate():
    # 384
    plate_384 = plate_generator.normal_plate(plate=384)
    assert plate_384.shape == SHAPE_384
    assert isinstance(plate_384.data, np.ndarray)
    # 1536
    plate_1536 = plate_generator.normal_plate(plate=1536)
    assert plate_1536.shape == SHAPE_1536
    assert isinstance(plate_1536.data, np.ndarray)


def test_uniform_plate():
    # 384
    plate_384 = plate_generator.uniform_plate(plate=384)
    assert plate_384.shape == SHAPE_384
    assert isinstance(plate_384.data, np.ndarray)
    # 1536
    plate_1536 = plate_generator.uniform_plate(plate=1536)
    assert plate_1536.shape == SHAPE_1536
    assert isinstance(plate_1536.data, np.ndarray)


def test_lognormal_plate():
    # 384
    plate_384 = plate_generator.lognormal_plate(plate=384)
    assert plate_384.shape == SHAPE_384
    assert isinstance(plate_384.data, np.ndarray)
    # 1536
    plate_1536 = plate_generator.lognormal_plate(plate=1536)
    assert plate_1536.shape == SHAPE_1536
    assert isinstance(plate_1536.data, np.ndarray)


def test_edge_plate():
    # 384
    plate_384 = plate_generator.edge_plate(plate=384)
    assert plate_384.shape == SHAPE_384
    assert isinstance(plate_384.data, np.ndarray)
    # 1536
    plate_1536 = plate_generator.edge_plate(plate=1536)
    assert plate_1536.shape == SHAPE_1536
    assert isinstance(plate_1536.data, np.ndarray)


def test_row_plate():
    # 384
    plate_384 = plate_generator.row_plate(plate=384)
    assert plate_384.shape == SHAPE_384
    assert isinstance(plate_384.data, np.ndarray)
    # 1536
    plate_1536 = plate_generator.row_plate(plate=1536)
    assert plate_1536.shape == SHAPE_1536
    assert isinstance(plate_1536.data, np.ndarray)


def test_column_plate():
    # 384
    plate_384 = plate_generator.column_plate(plate=384)
    assert plate_384.shape == SHAPE_384
    assert isinstance(plate_384.data, np.ndarray)
    # 1536
    plate_1536 = plate_generator.column_plate(plate=1536)
    assert plate_1536.shape == SHAPE_1536
    assert isinstance(plate_1536.data, np.ndarray)


def test_single_checker_plate():
    # 384
    plate_384 = plate_generator.single_checker_plate(plate=384)
    assert plate_384.shape == SHAPE_384
    assert isinstance(plate_384.data, np.ndarray)
    # 1536
    plate_1536 = plate_generator.single_checker_plate(plate=1536)
    assert plate_1536.shape == SHAPE_1536
    assert isinstance(plate_1536.data, np.ndarray)


def test_quad_checker_plate():
    # 384
    plate_384 = plate_generator.quad_checker_plate(plate=384)
    assert plate_384.shape == SHAPE_384
    assert isinstance(plate_384.data, np.ndarray)
    # 1536
    plate_1536 = plate_generator.quad_checker_plate(plate=1536)
    assert plate_1536.shape == SHAPE_1536
    assert isinstance(plate_1536.data, np.ndarray)


def test_h_grad_plate():
    # 384
    plate_384 = plate_generator.h_grad_plate(plate=384)
    assert plate_384.shape == SHAPE_384
    assert isinstance(plate_384.data, np.ndarray)
    # 1536
    plate_1536 = plate_generator.h_grad_plate(plate=1536)
    assert plate_1536.shape == SHAPE_1536
    assert isinstance(plate_1536.data, np.ndarray)


def test_v_grad_plate():
    # 384
    plate_384 = plate_generator.v_grad_plate(plate=384)
    assert plate_384.shape == SHAPE_384
    assert isinstance(plate_384.data, np.ndarray)
    # 1536
    plate_1536 = plate_generator.v_grad_plate(plate=1536)
    assert plate_1536.shape == SHAPE_1536
    assert isinstance(plate_1536.data, np.ndarray)


def test_bleedthrough_plate():
    # 384
    plate_384 = plate_generator.bleed_through_plate(plate=384)
    assert plate_384.shape == SHAPE_384
    assert isinstance(plate_384.data, np.ndarray)
    # 1536
    plate_1536 = plate_generator.bleed_through_plate(plate=1536)
    assert plate_1536.shape == SHAPE_1536
    assert isinstance(plate_1536.data, np.ndarray)


def test_snake_plate():
    # 384
    plate_384 = plate_generator.snake_plate(plate=384)
    assert plate_384.shape == SHAPE_384
    assert isinstance(plate_384.data, np.ndarray)
    # 1536
    plate_1536 = plate_generator.snake_plate(plate=1536)
    assert plate_1536.shape == SHAPE_1536
    assert isinstance(plate_1536.data, np.ndarray)


def test_add_plates():
    plate_a = plate_generator.normal_plate()
    plate_b = plate_generator.column_plate()
    combined_plate = plate_a + plate_b
    assert isinstance(combined_plate, plate_generator.Plate)
    # do the same operation in numpy
    combined_array = plate_a.data + plate_b.data
    assert (combined_array == combined_plate.data).all()
    # try adding a scalar value
    add_one = plate_a + 1
    assert isinstance(add_one, plate_generator.Plate)
    assert isinstance(add_one.data, np.ndarray)
    # to the same operation in numpy
    add_one_array = plate_a.data + 1
    assert (add_one.data == add_one_array).all()
    # add multiple
    plate_c = plate_generator.column_plate()
    combined_three = plate_a + plate_b + plate_c
    assert isinstance(combined_three.data, np.ndarray)
    assert isinstance(combined_three, plate_generator.Plate)


def test_multiply_plates():
    plate_a = plate_generator.normal_plate()
    plate_b = plate_generator.column_plate()
    combined_plate = plate_a * plate_b
    assert isinstance(combined_plate, plate_generator.Plate)
    # do the same operation in numpy
    combined_array = plate_a.data * plate_b.data
    assert (combined_array == combined_plate.data).all()
    # try adding a scalar value
    mult_two = plate_a * 2
    assert isinstance(mult_two, plate_generator.Plate)
    assert isinstance(mult_two.data, np.ndarray)
    # to the same operation in numpy
    mult_two_array = plate_a.data * 2
    assert (mult_two.data == mult_two_array).all()


def test_subtract_plates():
    plate_a = plate_generator.normal_plate()
    plate_b = plate_generator.column_plate()
    subtracted_plate = plate_a - plate_b
    assert isinstance(subtracted_plate, plate_generator.Plate)
    # do the same operation in numpy
    subtracted_array = plate_a.data - plate_b.data
    assert (subtracted_array == subtracted_plate.data).all()
    # try adding a scalar value
    subtract_one = plate_a - 1
    assert isinstance(subtract_one, plate_generator.Plate)
    assert isinstance(subtract_one.data, np.ndarray)
    # to the same operation in numpy
    subtract_one_array = plate_a.data - 1
    assert (subtract_one.data == subtract_one_array).all()


def test_divide_plates():
    plate_a = plate_generator.normal_plate()
    plate_b = plate_generator.column_plate()
    div_plate = plate_a / plate_b
    assert isinstance(div_plate, plate_generator.Plate)
    # do the same operation in numpy
    div_array = plate_a.data / plate_b.data
    assert (div_array == div_plate.data).all()
    # try adding a scalar value
    div_two = plate_a / 2
    assert isinstance(div_two, plate_generator.Plate)
    assert isinstance(div_two.data, np.ndarray)
    # to the same operation in numpy
    div_two_array = plate_a.data / 2
    assert (div_two.data == div_two_array).all()


def test_normalise_plates():
    plate = plate_generator.edge_plate()
    plate = plate + 10
    plate_norm = plate.normalise()
    # check the inital plate isn't normalised by accident
    assert abs(plate.data.mean()) > EPS
    assert abs(plate.data.std() - 1) > EPS
    # check the normalised plate has a mean of 0 and std of 1 
    assert abs(plate_norm.data.mean()) < EPS
    assert abs(plate_norm.data.std() - 1) < EPS


def test_normalise_inplace():
    plate = plate_generator.normal_plate()
    plate = plate + 10
    # check the inital plate isn't normalised by accident
    assert abs(plate.data.mean()) > EPS
    assert abs(plate.data.std() - 1) > EPS
    plate._normalise()
    # check the normalised plate has a mean of 0 and std of 1 
    assert abs(plate.data.mean()) < EPS
    assert abs(plate.data.std() - 1) < EPS


def test_mean():
    plate = plate_generator.normal_plate()
    output = plate.mean()
    np_output = plate.data.mean()
    assert isinstance(output, float)
    assert output == np_output


def test_std():
    plate = plate_generator.normal_plate()
    output = plate.std()
    np_output = plate.data.std()
    assert isinstance(output, float)
    assert output == np_output
