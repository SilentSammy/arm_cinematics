import math
import time
import numpy as np
import matplotlib.pyplot as plt
import help

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
    
    def get_pspace_pos(self):
        return self._x, self._y

    def get_cspace_pos(self, arm, wrap=True, as_degrees=True):
        if self._cspace is None:
            self._cspace = inverse_kinematics(self._x, self._y, arm.dh[0]['r'], arm.dh[1]['r'])
            if as_degrees:
                self._cspace = [(np.degrees(theta1), np.degrees(theta2)) for theta1, theta2 in self._cspace]
            if wrap:
                self._cspace = [(theta1 % 360, theta2 % 360) for theta1, theta2 in self._cspace]
        return tuple(self._cspace)

    def get_paths(self, arm):
        # Same as pathfind but returns all candidates
        current_angles = arm.get_cspace_pos(wrap=True, as_degrees=True)
        wrapped_goals = self.get_cspace_pos(arm)

        candidates = []
        for candidate in wrapped_goals:
            virtual_candidates = help.compute_all_quadrants(candidate, 360)
            for candidate_shifted in virtual_candidates:
                diff0 = candidate_shifted[0] - current_angles[0]
                diff1 = candidate_shifted[1] - current_angles[1]
                dist = np.hypot(diff0, diff1)
                candidates.append((dist, candidate_shifted))
        candidates.sort(key=lambda x: x[0])
        return candidates

    def get_path(self, arm, goal_index=0):
        # Now we can simply call self.get_paths(arm) to get the candidates
        candidates = self.get_paths(arm)

        # And select the goal_index-th candidate but with wrapping
        return candidates[goal_index % len(candidates)][1]

    def pathfind(self, arm, collisions):
        # get all paths
        candidates = self.get_paths(arm)

        # find the first collision-free path
        for _, candidate in candidates:
            if not Goal.path_collides(arm.get_cspace_pos(), candidate, collisions):
                return candidate

    @staticmethod
    def path_collides(start_cspace, target_cspace, collision_list, threshold=10.0):
        # By default, use the provided start and target.
        p_start = np.array(start_cspace)
        p_target = np.array(target_cspace)
        
        # If the target is off the central quadrant, compute the wrapped path.
        if help.get_quadrant(target_cspace, 360) != (0, 0):
            # Get the target's quadrant and compute its opposite.
            target_quad = help.get_quadrant(target_cspace, 360)
            opposite_quad = help.get_opposite_quadrant(target_quad)
            # Compute a shifted start so that the line goes through the central region.
            p_start = np.array(help.compute_quadrant(start_cspace, opposite_quad, 360))
            # Make the target centered in the central quadrant.
            p_target = np.array(help.compute_quadrant(target_cspace, (0, 0), 360))
        
        for collision in collision_list:
            collision_np = np.array(collision)
            dist = line_segment_distance(p_start, p_target, collision_np)
            if dist < threshold:
                return True
        return False

class Arm:
    def __init__(self, dh=None):
        self.dh = dh or [
            {'theta': math.radians(0), 'r': 1},
            {'theta': math.radians(0), 'r': 1},
            {'theta': math.radians(0), 'r': 1},
        ]

    def get_pspace_pos(self):
        """ Get the position of the end effector in the physical space. """
        x, y = 0, 0
        theta = 0
        for link in self.dh:
            theta += link['theta']
            x += link['r'] * np.cos(theta)
            y += link['r'] * np.sin(theta)
        return x, y

    def get_cspace_pos(self, wrap=True, as_degrees=True):
        """ Get the arm position in the configuration space. """
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
            x = last_x + link['r'] * math.cos(theta)
            y = last_y + link['r'] * math.sin(theta)
            positions.append((x, y))
        return positions

    def get_joint_angles(self):
        return tuple(link['theta'] for link in self.dh)

    def get_joint_ranges(self):
        # returns a list of x, y, and radius defining the joint ranges
        ranges = []
        positions = self.get_joint_positions()
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            radius = sum(link['r'] for link in self.dh[i:])
            ranges.append((x1, y1, radius))
        return ranges
    
    def scan_cspace(self, obstacle):
        collisions = []
        distances = [0 for _ in self.dh]
        joint_positions = self.get_joint_positions()

        def get_distance_to_obs(joint_idx):
            # get the distance from the segment to the obstacle
            line = (joint_positions[joint_idx], joint_positions[joint_idx + 1])
            seg_dist = max(help.distance_line_to_circle(line, obstacle), 0)
            distances[joint_idx] = seg_dist
            return seg_dist
        
        def is_within_range(joint_idx):
            # if previous joints are not moved, can this joint and subsequent ones collide with the obstacle?
            return help.distance_between_circles(self.get_joint_ranges()[joint_idx], obstacle) < 0

        def bulk_add_collisions(joint_idx):
            possible_angles = list(range(0, 360, 20))

            num_subsequent_joints = len(self.dh) - joint_idx

            # get all possible combinations of angles for the subsequent joints
            possible_joint_angles = help.all_possible_combinations(possible_angles, num_subsequent_joints)

            # insert the current joint's angle at the beginning of each combination
            collisions = [[self.dh[joint_idx]['theta']] + angles for angles in possible_joint_angles]

            # add the collisions to the list
            collisions.extend(collisions)

            # Get this and previous joint angles
            angles = self.get_joint_angles()[:joint_idx + 1]
            print(f"Joint {joint_idx} cascading collisions at angles:", [round(np.degrees(angle), 0) for angle in angles])

        def add_collision(joint_idx):
            collisions.append(self.get_joint_angles())
            print(f"Joint {joint_idx} colliding at angles:", [round(np.degrees(angle), 0) for angle in collisions[-1]])
        
        def get_next_step(joint_idx):
            # get the distance from the joint's origin to the obstacle
            origin = joint_positions[joint_idx]
            origin_dist = max(help.distance_point_to_circle(origin, obstacle), 0)

            seg_dist = distances[joint_idx]

            # get a normalized distance based on the origin's distance being 1
            dist = seg_dist / origin_dist if origin_dist > 0 else 0

            # at a dist of 1, step is 30, at 0, step is 10
            step = (dist * 20) + 10

            return step

        def check_collision(joint_idx):
            collision = get_distance_to_obs(joint_idx) <= 0
            if collision:
                if joint_idx == len(self.dh) - 1: # if it's a leaf joint
                    add_collision(joint_idx)
                else: # if it's not a leaf joint
                    bulk_add_collisions(joint_idx)

            return collision
        
        def set_angle(joint_idx, angle):
            nonlocal joint_positions
            self.dh[joint_idx]['theta'] = np.radians(angle)
            joint_positions = self.get_joint_positions()
        
        def get_angle(joint_idx):
            return np.degrees(self.dh[joint_idx]['theta'])

        def step_joint(joint_idx):
            angle = min(get_angle(joint_idx) + get_next_step(joint_idx), 360)
            set_angle(joint_idx, angle)

        def recursive_scan(joint_idx):
            if joint_idx >= self.num_joints:
                # Base case: all joints have been set. Yield final configuration.
                yield
                return

            # Initialize this joint angle to 0.
            set_angle(joint_idx, 0)
            # Continue adjusting this joint while less than 360.
            while get_angle(joint_idx) < 360:
                colliding = check_collision(joint_idx)
                # If no collision on this joint, and if there is a subsequent joint to scan,
                # and that subsequent joint is within range, drill down recursively.
                if not colliding:
                    if joint_idx < self.num_joints - 1 and is_within_range(joint_idx + 1):
                        yield from recursive_scan(joint_idx + 1)
                    # If at the last joint without collision, you might want to yield the config.
                else:
                    # If a collision is detected at a leaf joint, record it.
                    if joint_idx == self.num_joints - 1:
                        add_collision(joint_idx)
                    # For intermediate joints, you could optionally handle "bulk" collisions here.
                step_joint(joint_idx)
                yield  # Yield control after each step.

        # Start the recursive scan at joint 0.
        yield from recursive_scan(0)
        # Optionally, yield one final time after scanning is complete.
        return collisions
