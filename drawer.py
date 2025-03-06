import math
import time
import numpy as np
import matplotlib.pyplot as plt
from arm import Arm, Goal, line_segment_distance
import quadrant

class ArmDrawer:
    def __init__(self, arm, ax):
        self.arm = arm
        self.ax = ax
        self.point_artists = []     
        self.line_artists = []      
        self.range_artist = None    
        self.draw_range_circle()

    def draw_range_circle(self):
        max_reach = sum(link['r'] for link in self.arm.dh)
        circle = plt.Circle((0, 0), max_reach, color='r', fill=False, linestyle='--')
        self.ax.add_artist(circle)
        self.range_artist = circle

    def draw(self):
        positions = self.arm.get_joint_positions()
        for i, (x, y) in enumerate(positions):
            if i >= len(self.point_artists):
                artist, = self.ax.plot([x], [y], 'bo')
                self.point_artists.append(artist)
            else:
                self.point_artists[i].set_data([x], [y])
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
    def __init__(self, ax, arm, goal=None, obstacle=None):
        self.ax = ax
        self.arm_drawer = ArmDrawer(arm, ax)
        self.goal = goal or Goal(0, 0)
        self.obstacle = obstacle or (0, 0, 0.5)  # (x, y, radio)
        self.ax.set_title('Espacio Físico')
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.goal_point = ax.plot([self.goal.x], [self.goal.y], 'go')[0]
        self.obstacle_circle = plt.Circle((self.obstacle[0], self.obstacle[1]), 
                                          self.obstacle[2], color='red', alpha=0.5)
        self.ax.add_patch(self.obstacle_circle)
        self.draw()

    def draw(self):
        self.arm_drawer.draw()
        self.goal_point.set_data([self.goal.x], [self.goal.y])

class CSpaceDrawer:
    def __init__(self, ax, arm, goal, obstacle):
        self.ax = ax
        self.arm: Arm = arm
        self.goal = goal
        self.obstacle = obstacle
        self.ax.set_title('Espacio de Configuraciones (3D)')
        self.ax.set_xlim(0, 360)
        self.ax.set_ylim(0, 360)
        self.ax.set_zlim(0, 360)
        self.ax.set_xlabel("Ángulo 1 (°)")
        self.ax.set_ylabel("Ángulo 2 (°)")
        self.ax.set_zlabel("Ángulo 3 (°)")
        self.end_point, = ax.plot([0], [0], [0], 'bo')
        self.goal_points = []
        for p in self.goal.get_cspace_pos(arm):
            gp, = ax.plot([p[0]], [p[1]], [p[2]], 'go')
            self.goal_points.append(gp)
        self.arm_to_goal_line, = ax.plot([0, 0], [0, 0], [0, 0], 'k--')
        self.goal_to_arm_line, = ax.plot([0, 0], [0, 0], [0, 0], 'k--')
        self.collisions = []
        self.collision_scatter = ax.scatter([], [], [], c='r')
        self.closest_goal = None

    def draw(self):
        end_eff = self.arm.get_cspace_pos(wrap=True, as_degrees=True)
        self.end_point.set_data([end_eff[0]], [end_eff[1]])
        self.end_point.set_3d_properties([end_eff[2]])
        goals = self.goal.get_cspace_pos(self.arm)
        for i, pt in enumerate(goals):
            self.goal_points[i].set_data([pt[0]], [pt[1]])
            self.goal_points[i].set_3d_properties([pt[2]])
        self.draw_path(end_eff)
        self.draw_collisions()

    def draw_path(self, end_eff):
        closest_goal = self.closest_goal
        if closest_goal is None:
            self.arm_to_goal_line.set_data([], [])
            self.arm_to_goal_line.set_3d_properties([])
            self.goal_to_arm_line.set_data([], [])
            self.goal_to_arm_line.set_3d_properties([])
            return
        
        collides = False  # Se puede ampliar la verificación
        line_color = 'r' if collides else 'k'
        self.arm_to_goal_line.set_color(line_color)
        self.goal_to_arm_line.set_color(line_color)

        self.arm_to_goal_line.set_data([end_eff[0], closest_goal[0]], [end_eff[1], closest_goal[1]])
        self.arm_to_goal_line.set_3d_properties([end_eff[2], closest_goal[2]])

        if quadrant.get_quadrant(closest_goal, 360) != (0, 0, 0):
            quadrant_goal = quadrant.get_quadrant(closest_goal, 360)
            opposite_quadrant = quadrant.get_opposite_quadrant(quadrant_goal)
            opposite_end = quadrant.compute_quadrant(end_eff, opposite_quadrant, 360)
            centered_goal = quadrant.compute_quadrant(closest_goal, (0, 0, 0), 360)
            self.goal_to_arm_line.set_data([opposite_end[0], centered_goal[0]], [opposite_end[1], centered_goal[1]])
            self.goal_to_arm_line.set_3d_properties([end_eff[2], closest_goal[2]])
        else:
            self.goal_to_arm_line.set_data([], [])
            self.goal_to_arm_line.set_3d_properties([])

    def draw_collisions(self):
        if self.collisions:
            x_data, y_data, z_data = zip(*self.collisions)
            self.collision_scatter._offsets3d = (x_data, y_data, z_data)

    def scan_generator(self):
        self.collisions.clear()
        increment = 10  
        for i in range(0, 360, increment):
            for j in range(0, 360, increment):
                for k in range(0, 360, increment):
                    self.arm.dh[0]['theta'] = np.radians(i)
                    self.arm.dh[1]['theta'] = np.radians(j)
                    self.arm.dh[2]['theta'] = np.radians(k)
                    pos = self.arm.get_joint_positions()
                    dist1 = line_segment_distance(np.array(pos[0]), np.array(pos[1]), np.array(self.obstacle[:2]))
                    dist2 = line_segment_distance(np.array(pos[1]), np.array(pos[2]), np.array(self.obstacle[:2]))
                    dist3 = line_segment_distance(np.array(pos[2]), np.array(pos[3]), np.array(self.obstacle[:2]))
                    collision1 = dist1 < self.obstacle[2]
                    collision2 = dist2 < self.obstacle[2]
                    collision3 = dist3 < self.obstacle[2]
                    if collision1 or collision2 or collision3:
                        self.collisions.append((i, j, k))
                    yield
        self.draw_collisions()
        yield

    def scan(self):
        for _ in self.scan_generator():
            pass
