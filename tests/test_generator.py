"""
module docstring
"""

from plate_generator import generator
from plate_generator.core import Plate

def test_generator():
    my_generator = generator.generator(n=10)
    for plate, name in my_generator:
        assert isinstance(plate, Plate)
