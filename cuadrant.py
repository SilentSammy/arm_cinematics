import numpy as np

def compute_cuadrant(point, cuadrant, size):
    # Calculate the point in the central cuadrant
    wrapped_point = tuple(v % size for v in point)

    # Calculate the offset to apply to the point
    offset = tuple(v * size for v in cuadrant)

    # Calculate the point in the new cuadrant
    new_point = tuple(wrapped_point[i] + offset[i] for i in range(len(point)))
    return new_point

def compute_all_cuadrants(point, size):
    # All possible pairs of -1, 0 and 1
    cuadrants = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]

    # Calculate the point in all cuadrants
    return [compute_cuadrant(point, cuadrant, size) for cuadrant in cuadrants]

def get_cuadrant(point, size):
    # Calculate the point in the central cuadrant
    wrapped_point = tuple(v % size for v in point)

    # Calculate the cuadrant
    cuadrant = tuple(v // size for v in point)
    return cuadrant

def get_opposite_cuadrant(cuadrant):
    return tuple(-v for v in cuadrant)