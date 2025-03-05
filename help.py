import numpy as np

def all_possible_combinations(lst, n):
    if n == 0:
        return [[]]
    return [[x] + combo for x in lst for combo in all_possible_combinations(lst, n - 1)]

def line_segment_distance(p1, p2, p):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p = np.array(p)
    v = p2 - p1
    w = p - p1
    vv = np.dot(v, v)
    if vv < 1e-9:
        return np.linalg.norm(p - p1)
    t = np.dot(w, v) / vv
    if t < 0.0:
        return np.linalg.norm(p - p1)
    elif t > 1.0:
        return np.linalg.norm(p - p2)
    else:
        proj = p1 + t * v
        return np.linalg.norm(p - proj)

def distance_between_circles(c1, c2):
    # given two circles (x, y, r), return the distance between their edges
    return np.linalg.norm(np.array(c1[:2]) - np.array(c2[:2])) - c1[2] - c2[2]

def distance_point_to_circle(p, c):
    # given a point (x, y) and a circle (x, y, r), return the distance between the point and the circle's edge
    return np.linalg.norm(np.array(p) - np.array(c[:2])) - c[2]

def distance_line_to_circle(l, c):
    # given a line segment ((x1, y1), (x2, y2)) and a circle (x, y, r), return the distance between the line and the circle's edge
    dist_to_center = line_segment_distance(l[0], l[1], c[:2])
    return dist_to_center - c[2]

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
