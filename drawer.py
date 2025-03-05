import math
import time
import numpy as np
import matplotlib.pyplot as plt
from arm import Arm, Goal
import help
from mpl_toolkits.mplot3d import Axes3D

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
    def __init__(self, fig, arm, goal, obstacle):
        # Objects to draw
        self.fig = fig
        self.arm: Arm = arm
        self.goal = goal
        self.obstacle = obstacle
        self.dimensions = self.arm.num_joints

        # Setup ax
        if self.dimensions == 2:
            self.ax = fig.add_subplot(1, 2, 2)
            self.ax.set_title('C-space')
            self.ax.set_xlim(0, 360)
            self.ax.set_ylim(0, 360)
            self.ax.set_xlabel("Joint 1 Angle (degrees)")
            self.ax.set_ylabel("Joint 2 Angle (degrees)")
        elif self.dimensions == 3:
            self.ax = fig.add_subplot(1, 2, 2, projection='3d')
            self.ax.set_title('C-space')
            self.ax.set_xlim(0, 360)
            self.ax.set_ylim(0, 360)
            self.ax.set_zlim(0, 360)
            self.ax.set_xlabel("Joint 1 Angle (degrees)")
            self.ax.set_ylabel("Joint 2 Angle (degrees)")
            self.ax.set_zlabel("Joint 3 Angle (degrees)")

        # Create artists
        if self.dimensions == 2:
            self.end_point = self.ax.plot(0, 0, 'bo')[0]  # end effector position in C-space
            self.goal_points = []  # goal positions in C-space
            # for p in goal.get_cspace_pos(arm):
            #     self.goal_points.append(ax.plot(p[0], p[1], 'go')[0])
        elif self.dimensions == 3:
            self.end_point = self.ax.scatter(0, 0, 0, c='b', marker='o')  # end effector position in C-space
            self.goal_points = []  # goal positions in C-space
            # for p in goal.get_cspace_pos(arm):
            #     self.goal_points.append(ax.scatter(p[0], p[1], p[2], c='g', marker='o'))

        # Create connection lines from end effector to each goal
        self.closest_goal = None
        if self.dimensions == 2:
            self.arm_to_goal_line = self.ax.plot([], [], 'k--')[0]
            self.goal_to_arm_line = self.ax.plot([], [], 'k--')[0]
        elif self.dimensions == 3:
            self.arm_to_goal_line = self.ax.plot([], [], [], 'k--')[0]
            self.goal_to_arm_line = self.ax.plot([], [], [], 'k--')[0]

        # Collision points in C-space
        self.collisions = []
        if self.dimensions == 2:
            self.collision_scatter = self.ax.scatter([], [], c='r')  # scatter plot for collision points
        elif self.dimensions == 3:
            self.collision_scatter = self.ax.scatter([], [], [], c='r')  # scatter plot for collision points

    def draw(self):
        # Get current arm angles (in degrees)
        end_eff = self.arm.get_cspace_pos(wrap=True, as_degrees=True)

        if self.dimensions == 2:
            self.end_point.set_data([end_eff[0]], [end_eff[1]])
        elif self.dimensions == 3:
            self.end_point._offsets3d = ([end_eff[0]], [end_eff[1]], [end_eff[2]])

        # Update IK goal markers
        # goals = self.goal.get_cspace_pos(self.arm)
        # for i, pt in enumerate(goals):
        #     self.goal_points[i].set_data([pt[0]], [pt[1]])

        # self.draw_path(end_eff)

        # self.draw_collisions()

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
            x_data = np.degrees(x_data)
            y_data = np.degrees(y_data)
            self.collision_scatter.set_offsets(np.c_[x_data, y_data])