import math
import time
import numpy as np
import matplotlib.pyplot as plt
from arm import Arm, Goal
import help

class ArmDrawer:
    def __init__(self, arm, ax):
        self.arm = arm
        self.ax = ax
        self.point_artists = []  # to hold the plotted joint points
        self.line_artists = []   # to hold the connecting line segments
        self.joint_range_artists = []

    def draw_joint_range_circles(self):
        """
        Draw circles around each joint's origin representing the maximum reach from that joint.
        """
        ranges = self.arm.get_joint_ranges() # this returns points with radiuses

        # Draw the circles with dotted lines
        for i, (x, y, r) in enumerate(ranges):
            if i >= len(self.joint_range_artists):
                artist = plt.Circle((x, y), r, color='b', fill=False, linestyle='dotted')
                self.ax.add_patch(artist)
                self.joint_range_artists.append(artist)
            else:
                self.joint_range_artists[i].center = (x, y)
                self.joint_range_artists[i].radius = r

    def draw(self):
        """
        Redraw the arm based on the Arm's current joint positions.
        """
        positions = list(self.arm.get_joint_positions())
        
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
        
        # Draw joint range circles
        self.draw_joint_range_circles()
    
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
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
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
        self.goal_points = []  # goal positions in C-space
        for p in goal.get_cspace_pos(arm):
            self.goal_points.append(ax.plot(p[0], p[1], 'go')[0])
        
        # Create connection lines from end effector to each goal
        self.closest_goal = None
        self.arm_to_goal_line = ax.plot([], [], 'k--')[0]
        self.goal_to_arm_line = ax.plot([], [], 'k--')[0]
        
        # Collision points in C-space
        self.collisions = []
        self.collision_scatter = self.ax.scatter([], [], c='r')  # scatter plot for collision points

    def draw(self):
        # Get current arm angles (in degrees)
        end_eff = self.arm.get_cspace_pos(wrap=True, as_degrees=True)
        self.end_point.set_data([end_eff[0]], [end_eff[1]])

        # Update IK goal markers
        goals = self.goal.get_cspace_pos(self.arm)
        for i, pt in enumerate(goals):
            self.goal_points[i].set_data([pt[0]], [pt[1]])

        self.draw_path(end_eff)

        self.draw_collisions()

    def draw_path(self, end_eff):
        # Get the user-selected goal
        closest_goal = self.closest_goal
        if closest_goal is None:
            self.arm_to_goal_line.set_data([], [])
            self.goal_to_arm_line.set_data([], [])
            return
        
        # Determine if the path to the goal collides
        # collides = Goal.path_collides(end_eff, closest_goal, self.collisions)
        collides = False # Disabled to save resources
        print("Path collides!" if collides else "Path clear!")
        
        # Set color
        line_color = 'r' if collides else 'k'
        self.arm_to_goal_line.set_color(line_color)
        self.goal_to_arm_line.set_color(line_color)

        # Draw the direct connection line
        self.arm_to_goal_line.set_data([end_eff[0], closest_goal[0]], [end_eff[1], closest_goal[1]])

        # if the goal is not in the central quadrant, draw a line coming from the opposite quadrant
        if help.get_quadrant(closest_goal, 360) != (0, 0):
            quadrant_goal = help.get_quadrant(closest_goal, 360) # Get the quadrant of the closest goal
            opposite_quadrant = help.get_opposite_quadrant(quadrant_goal) # Get the opposite quadrant
            opposite_end = help.compute_quadrant(end_eff, opposite_quadrant, 360) # Get the position of the end effector in the opposite quadrant
            centered_goal = help.compute_quadrant(closest_goal, (0, 0), 360) # Get the centered goal
            self.goal_to_arm_line.set_data([opposite_end[0], centered_goal[0]], [opposite_end[1], centered_goal[1]]) # Draw the wrapping line
        else:
            self.goal_to_arm_line.set_data([], [])
        

    def draw_collisions(self):
        if self.collisions:
            x_data, y_data = zip(*self.collisions)
            self.collision_scatter.set_offsets(np.c_[x_data, y_data])

    def scan_generator(self):
        """
        Generator version of scan that yields control after each inner loop iteration.
        The external loop can step through and animate each frame as desired.
        """
        
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
                pos = list(self.arm.get_joint_positions())

                # Get distances for the two segments
                dist1 = help.line_segment_distance(np.array(pos[0]), np.array(pos[1]), np.array(self.obstacle[:2]))
                dist2 = help.line_segment_distance(np.array(pos[1]), np.array(pos[2]), np.array(self.obstacle[:2]))
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
        for _ in self.scan_generator():
            pass
