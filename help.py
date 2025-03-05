import numpy as np

def distance_between_circles(c1, c2):
    # given two circles (x, y, r), return the distance between their edges
    return np.linalg.norm(np.array(c1[:2]) - np.array(c2[:2])) - c1[2] - c2[2]

def compute_quadrant(point, quadrant, size):
    # Calculate the point in the central quadrant
    wrapped_point = tuple(v % size for v in point)

    # Calculate the offset to apply to the point
    offset = tuple(v * size for v in quadrant)

    # Calculate the point in the new quadrant
    new_point = tuple(wrapped_point[i] + offset[i] for i in range(len(point)))
    return new_point

def compute_all_quadrants(point, size):
    # All possible pairs of -1, 0 and 1
    quadrants = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]

    # Calculate the point in all quadrants
    return [compute_quadrant(point, quadrant, size) for quadrant in quadrants]

def get_quadrant(point, size):
    # Calculate the point in the central quadrant
    wrapped_point = tuple(v % size for v in point)

    # Calculate the quadrant
    quadrant = tuple(v // size for v in point)
    return quadrant

def get_opposite_quadrant(quadrant):
    return tuple(-v for v in quadrant)