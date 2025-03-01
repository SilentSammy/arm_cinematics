import math
import time
import numpy as np
import matplotlib.pyplot as plt
from arm import Arm, ArmDrawer, Goal

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
        self.arm:Arm = arm
        self.goal = goal
        self.obstacle = obstacle

        # Setup ax
        self.ax.set_title('C-space')
        self.ax.set_xlim(0, 360)
        self.ax.set_ylim(0, 360)
        self.ax.set_xlabel("Joint 1 Angle (degrees)")
        self.ax.set_ylabel("Joint 2 Angle (degrees)")

        # Create artists
        self.arm_pos = ax.plot(0, 0, 'bo')[0]  # plot the current position in C-space
        self.goal_point_c = []
        for p in goal.get_pos_cspace(arm):
            self.goal_point_c.append(ax.plot(p[0], p[1], 'go')[0])
        
    def draw(self):
        angles = self.arm.get_joint_angles()
        
        # draw the current arm position in C-space
        self.arm_pos.set_data([angles[0]], [angles[1]])

        # plot the goal position in C-space
        c_space_points = self.goal.get_pos_cspace(self.arm)
        for i in range(len(c_space_points)):
            self.goal_point_c[i].set_data([c_space_points[i][0]], [c_space_points[i][1]])
