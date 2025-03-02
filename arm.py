import time
import numpy as np
import matplotlib.pyplot as plt

def inverse_kinematics(x, y, l1, l2):
    """
    Calculate the inverse kinematics for a 2-jointed arm.
    Returns two sets of joint angles (theta1, theta2).
    """
    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    sin_theta2 = np.sqrt(1 - cos_theta2**2)
    
    theta2_1 = np.arctan2(sin_theta2, cos_theta2)
    theta2_2 = np.arctan2(-sin_theta2, cos_theta2)
    
    k1 = l1 + l2 * cos_theta2
    k2_1 = l2 * sin_theta2
    k2_2 = -l2 * sin_theta2
    
    theta1_1 = np.arctan2(y, x) - np.arctan2(k2_1, k1)
    theta1_2 = np.arctan2(y, x) - np.arctan2(k2_2, k1)
    
    return (theta1_1, theta2_1), (theta1_2, theta2_2)

class Goal:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._cspace = None
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        if self._x != value:
            self._x = value
            self._cspace = None
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        if self._y != value:
            self._y = value
            self._cspace = None
    
    def get_pos_cspace(self, arm):
        if self._cspace is None:
            self._cspace = inverse_kinematics(self._x, self._y, arm.dh[0]['r'], arm.dh[1]['r'])
            self._cspace = [(np.degrees(theta1) % 360, np.degrees(theta2) % 360) for theta1, theta2 in self._cspace]
        return self._cspace

    def pathfind(self, arm, goal_index=0):
        # Get the current joint angles (in degrees), not wrapped to reflect the true position.
        current_angles = arm.get_joint_angles(wrap=True, as_degrees=True)
        
        # Get the wrapped goals (each in range 0-360) computed from inverse kinematics.
        wrapped_goals = self.get_pos_cspace(arm)
        
        candidates = []
        # Define candidate shifts for each joint:
        shifts = [-360, 0, 360]
        for candidate in wrapped_goals:
            # Generate candidate solutions for the candidate in all quadrants.
            for shift0 in shifts:
                for shift1 in shifts:
                    # Candidate plus shift vector
                    candidate_shifted = (candidate[0] + shift0, candidate[1] + shift1)
                    # Compute Euclidean distance between candidate and current arm configuration.
                    diff0 = candidate_shifted[0] - current_angles[0]
                    diff1 = candidate_shifted[1] - current_angles[1]
                    dist = np.hypot(diff0, diff1)
                    candidates.append((dist, candidate_shifted))
        
        # Sort the candidate solutions based on distance.
        candidates.sort(key=lambda x: x[0])
        
        # Choose one candidate using the goal_index parameter.
        chosen_candidate = candidates[goal_index % len(candidates)][1]
        return chosen_candidate

class Arm:
    def __init__(self, dh=None):
        self.dh = dh or [
            {'theta': np.radians(0), 'r': 1},
            {'theta': np.radians(0), 'r': 1},
        ]

    def get_end_effector_position(self):
        x, y = 0, 0
        theta = 0
        for link in self.dh:
            theta += link['theta']
            x += link['r'] * np.cos(theta)
            y += link['r'] * np.sin(theta)
        return x, y

    def get_joint_angles(self, wrap=True, as_degrees=True):
        angles = [link['theta'] for link in self.dh]
        if as_degrees:
            angles = [np.degrees(angle) for angle in angles]
        if wrap:
            angles = [angle % 360 for angle in angles]
        return angles

    @property
    def num_joints(self):
        return len(self.dh)

    def get_joint_positions(self):
        positions = [(0, 0)]  # start at origin
        theta = 0
        for link in self.dh:
            theta += link['theta']  # accumulate angle
            last_x, last_y = positions[-1]
            x = last_x + link['r'] * np.cos(theta)
            y = last_y + link['r'] * np.sin(theta)
            positions.append((x, y))
        return positions


