import numpy as np
from itertools import product

def compute_quadrant(point, quadrant, size):
    # Calculate the point in the central quadrant
    wrapped_point = tuple(v % size for v in point)

    # Calculate the offset to apply to the point
    offset = tuple(v * size for v in quadrant)

    # Calculate the point in the new quadrant
    new_point = tuple(wrapped_point[i] + offset[i] for i in range(len(point)))
    return new_point

def compute_all_quadrants(point, size):
    dims = len(point)
    cuadrants = list(product(range(-1, 2), repeat=dims))
    return [compute_quadrant(point, quadrant, size) for quadrant in cuadrants]

def get_quadrant(point, size):
    # Calculate the point in the central quadrant
    wrapped_point = tuple(v % size for v in point)

    # Calculate the quadrant
    quadrant = tuple(v // size for v in point)
    return quadrant

def get_opposite_quadrant(quadrant):
    return tuple(-v for v in quadrant)