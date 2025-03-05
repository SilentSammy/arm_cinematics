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
        self.point_artists = []     # para guardar los puntos de las articulaciones
        self.line_artists = []      # para guardar las líneas que conectan las articulaciones
        self.range_artist = None    # para dibujar el círculo de alcance máximo

        # Dibujar el círculo que representa el alcance máximo del brazo
        self.draw_range_circle()

    def draw_range_circle(self):
        max_reach = sum(link['r'] for link in self.arm.dh)
        circle = plt.Circle((0, 0), max_reach, color='r', fill=False, linestyle='--')
        self.ax.add_artist(circle)
        self.range_artist = circle

    def draw(self):
        positions = self.arm.get_joint_positions()
        # Dibujar o actualizar los puntos de las articulaciones
        for i, (x, y) in enumerate(positions):
            if i >= len(self.point_artists):
                artist, = self.ax.plot([x], [y], 'bo')
                self.point_artists.append(artist)
            else:
                self.point_artists[i].set_data([x], [y])
        # Dibujar o actualizar las líneas entre puntos consecutivos
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

        # Configuración del eje físico
        self.ax.set_title('Espacio Físico')
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        # Crear artistas para el objetivo y obstáculo
        self.goal_point = ax.plot([self.goal.x], [self.goal.y], 'go')[0]
        self.obstacle_circle = plt.Circle((self.obstacle[0], self.obstacle[1]), 
                                          self.obstacle[2], color='red', alpha=0.5)
        self.ax.add_patch(self.obstacle_circle)

        self.draw()

    def draw(self):
        self.arm_drawer.draw()
        self.goal_point.set_data([self.goal.x], [self.goal.y])

# --- Modificaciones para el espacio de configuraciones de un robot 3R ---
class CSpaceDrawer:
    def __init__(self, ax, arm, goal, obstacle):
        self.ax = ax
        self.arm: Arm = arm
        self.goal = goal
        self.obstacle = obstacle

        # Configurar el eje 3D para el C-space
        self.ax.set_title('Espacio de Configuraciones (3D)')
        self.ax.set_xlim(0, 360)
        self.ax.set_ylim(0, 360)
        self.ax.set_zlim(0, 360)
        self.ax.set_xlabel("Ángulo 1 (°)")
        self.ax.set_ylabel("Ángulo 2 (°)")
        self.ax.set_zlabel("Ángulo 3 (°)")

        # Crear el punto que representa la configuración actual del brazo
        self.end_point, = ax.plot([0], [0], [0], 'bo')
        # Crear puntos para las posibles soluciones IK del objetivo
        self.goal_points = []
        for p in self.goal.get_cspace_pos(arm):
            gp, = ax.plot([p[0]], [p[1]], [p[2]], 'go')
            self.goal_points.append(gp)
        
        # Líneas de conexión entre la configuración actual y el objetivo IK
        self.arm_to_goal_line, = ax.plot([0, 0], [0, 0], [0, 0], 'k--')
        self.goal_to_arm_line, = ax.plot([0, 0], [0, 0], [0, 0], 'k--')
        
        # Puntos de colisión en el C-space
        self.collisions = []
        self.collision_scatter = ax.scatter([], [], [], c='r')

        self.closest_goal = None  # Se asigna la solución IK seleccionada

    def draw(self):
        # Obtener la configuración actual en grados (3 ángulos)
        end_eff = self.arm.get_cspace_pos(wrap=True, as_degrees=True)
        self.end_point.set_data([end_eff[0]], [end_eff[1]])
        self.end_point.set_3d_properties([end_eff[2]])

        # Actualizar los marcadores de las soluciones IK del objetivo
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
        
        collides = False  # Aquí se podría evaluar la colisión de la trayectoria
        line_color = 'r' if collides else 'k'
        self.arm_to_goal_line.set_color(line_color)
        self.goal_to_arm_line.set_color(line_color)

        # Línea directa desde la configuración actual hasta la solución IK seleccionada
        self.arm_to_goal_line.set_data([end_eff[0], closest_goal[0]], [end_eff[1], closest_goal[1]])
        self.arm_to_goal_line.set_3d_properties([end_eff[2], closest_goal[2]])

        # Si el objetivo no se encuentra en el cuadrante central (en proyección 2D),
        # se puede dibujar una línea "envuelta" usando las funciones de quadrant.
        if quadrant.get_quadrant(closest_goal, 360) != (0, 0, 0):
            quadrant_goal = quadrant.get_quadrant(closest_goal, 360)
            opposite_quadrant = quadrant.get_opposite_quadrant(quadrant_goal)
            opposite_end = quadrant.compute_quadrant(end_eff, opposite_quadrant, 360)
            centered_goal = quadrant.compute_quadrant(closest_goal, (0, 0, 0), 360)
            self.goal_to_arm_line.set_data([opposite_end[0], centered_goal[0]], [opposite_end[1], centered_goal[1]])
            # Para la componente z, usamos los valores originales
            self.goal_to_arm_line.set_3d_properties([end_eff[2], closest_goal[2]])
        else:
            self.goal_to_arm_line.set_data([], [])
            self.goal_to_arm_line.set_3d_properties([])

    def draw_collisions(self):
        if self.collisions:
            # Separar los tres ángulos de cada punto de colisión
            x_data, y_data, z_data = zip(*self.collisions)
            # Actualizar el scatter 3D (usando la propiedad interna _offsets3d)
            self.collision_scatter._offsets3d = (x_data, y_data, z_data)

    def scan_generator(self):
        """
        Generador para escanear el espacio de configuraciones. Se itera en los tres ángulos (i, j, k)
        usando un incremento fijo (por ejemplo, 10°) para simplificar. En cada iteración se asignan
        los tres ángulos al brazo, se calcula la posición de las articulaciones y se verifica la colisión.
        """
        self.collisions.clear()
        increment = 10  # Incremento en grados (ajustable según necesidad)
        for i in range(0, 360, increment):
            for j in range(0, 360, increment):
                for k in range(0, 360, increment):
                    self.arm.dh[0]['theta'] = np.radians(i)
                    self.arm.dh[1]['theta'] = np.radians(j)
                    self.arm.dh[2]['theta'] = np.radians(k)
                    pos = self.arm.get_joint_positions()
                    # Calcular la distancia mínima entre cada segmento y el obstáculo (usando solo la proyección en x,y)
                    dist1 = line_segment_distance(np.array(pos[0]), np.array(pos[1]), np.array(self.obstacle[:2]))
                    dist2 = line_segment_distance(np.array(pos[1]), np.array(pos[2]), np.array(self.obstacle[:2]))
                    dist3 = line_segment_distance(np.array(pos[2]), np.array(pos[3]), np.array(self.obstacle[:2]))
                    collision1 = dist1 < self.obstacle[2]
                    collision2 = dist2 < self.obstacle[2]
                    collision3 = dist3 < self.obstacle[2]
                    if collision1 or collision2 or collision3:
                        self.collisions.append((i, j, k))
                    yield  # Permite actualizar/animar entre iteraciones
        self.draw_collisions()
        yield

    def scan(self):
        for _ in self.scan_generator():
            pass
