import math
import time
import numpy as np
import matplotlib.pyplot as plt
from arm import Arm, Goal

class ArmDrawer:
    def __init__(self, arm, ax):
        self.arm = arm
        self.ax = ax
        self.point_artists = []  # to hold the plotted joint points
        self.line_artists = []   # to hold the connecting line segments
        self.range_artist = None  # to hold the range circle

        # Draw the maximum movement range of the arm
        self.draw_range_circle()

    def draw_range_circle(self):
        """
        Draw a circle representing the maximum reach of the arm.
        """
        max_reach = sum(link['r'] for link in self.arm.dh)
        circle = plt.Circle((0, 0), max_reach, color='r', fill=False, linestyle='--')
        self.ax.add_artist(circle)
        self.range_artist = circle

    def draw(self):
        """
        Redraw the arm based on the Arm's current joint positions.
        """
        positions = self.arm.get_joint_positions()
        
        # Draw or update joint points
        for i, (x, y) in enumerate(positions):
            if i >= len(self.point_artists):
                artist, = self.ax.plot([x], [y], 'bo')
                self.point_artists.append(artist)
            else:
                self.point_artists[i].set_data([x], [y])
        
        # Draw or update lines between consecutive points
        # There will be one fewer line than points.
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i+1]
            if i >= len(self.line_artists):
                line_artist, = self.ax.plot([x1, x2], [y1, y2], 'k-')
                self.line_artists.append(line_artist)
            else:
                self.line_artists[i].set_data([x1, x2], [y1, y2])
    
    def get_artists(self):
        return self.point_artists + self.line_artists + [self.range_artist]

class PSpaceDrawer:
    def __init__(self, ax, arm, goal = None, obstacle = None):
        # Objects to draw
        self.ax = ax
        self.arm_drawer = ArmDrawer(arm, ax)
        self.goal = goal or Goal(0, 0)
        self.obstacle = obstacle or (0, 0, 0.5)  # x, y, radius

        # Setup ax
        self.ax.set_title('Physical Space')
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        # Create artists
        self.goal_point = ax.plot([goal.x], [goal.y], 'go')[0]  # plot the goal position in physical space
        self.obstacle_circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', alpha=0.5)
        self.ax.add_patch(self.obstacle_circle)

        self.draw()

    def draw(self):
        # Draw the arm in physical space
        self.arm_drawer.draw()

        # plot the goal position in physical space
        self.goal_point.set_data([self.goal.x], [self.goal.y])

class CSpaceDrawer:
    def __init__(self, ax, arm, goal, obstacle):
        # Objects to draw
        self.ax = ax
        self.arm: Arm = arm
        self.goal = goal
        self.obstacle = obstacle

        # Setup ax
        self.ax.set_title('C-space')
        self.ax.set_xlim(0, 360)
        self.ax.set_ylim(0, 360)
        self.ax.set_xlabel("Joint 1 Angle (degrees)")
        self.ax.set_ylabel("Joint 2 Angle (degrees)")

        # Create artists
        self.end_point = ax.plot(0, 0, 'bo')[0]  # end effector position in C-space
        self.goal_points = [] # goal positions in C-space
        for p in goal.get_pos_cspace(arm):
            self.goal_points.append(ax.plot(p[0], p[1], 'go')[0])
        self.collisions = [] # collision points in C-space
        self.collision_scatter = self.ax.scatter([], [], c='r')  # scatter plot for collision points

    def draw(self):
        angles = self.arm.get_joint_angles(wrap=True, as_degrees=True)
        
        # Update the current arm position marker in C-space
        self.end_point.set_data([angles[0]], [angles[1]])

        # Update goal markers based on inverse kinematics
        c_space_points = self.goal.get_pos_cspace(self.arm)
        for i in range(len(c_space_points)):
            self.goal_points[i].set_data([c_space_points[i][0]], [c_space_points[i][1]])

        # Draw collision points
        self.draw_collisions()

    def draw_collisions(self):
        if self.collisions:
            x_data, y_data = zip(*self.collisions)
            self.collision_scatter.set_offsets(np.c_[x_data, y_data])

    def scan_generator(self):
        """
        Generator version of scan that yields control after each inner loop iteration.
        The external loop can step through and animate each frame as desired.
        """
        def line_segment_distance(p1, p2, p):
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
        
        def add_point(point):
            self.collisions.append(point)
        
        self.collisions.clear()
        i = 0
        while i < 360:
            j = 0
            closest_dist = None
            while j < 360:
                self.arm.dh[0]['theta'] = np.radians(i)
                self.arm.dh[1]['theta'] = np.radians(j)
                pos = self.arm.get_joint_positions()

                # Get distances for the two segments
                dist1 = line_segment_distance(np.array(pos[0]), np.array(pos[1]), np.array(self.obstacle[:2]))
                dist2 = line_segment_distance(np.array(pos[1]), np.array(pos[2]), np.array(self.obstacle[:2]))
                dist = min(dist1, dist2)
                closest_dist = dist if closest_dist is None else min(closest_dist, dist)

                # Determine collision status
                collision1 = dist1 < self.obstacle[2]
                collision2 = dist2 < self.obstacle[2]
                collision = collision1 or collision2
                print(f"Joint 1 Angle: {i}, Joint 2 Angle: {j}, "
                      f"Distance 1: {dist1}, Distance 2: {dist2}",
                      "Collision!" if collision else "")
                
                if collision:
                    add_point((i, j))
                
                # After each inner loop iteration, yield control.
                yield  # external code may now update/redraw
                
                # Adjust increments
                increment2 = min(max(10, int(dist2 * 30)), 90)
                j += increment2
            increment1 = min(max(5, int(closest_dist * 20)), 90)
            i += increment1
        
        # At the end, yield one last time after drawing all collisions.
        self.draw_collisions()
        yield

    def scan(self):
        """
        Scan the C-space for collisions between the arm and the obstacle.
        """
        for _ in self.scan_generator():
            pass
