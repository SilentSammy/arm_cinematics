import time
import numpy as np
import matplotlib.pyplot as plt
import quadrant

import numpy as np

def inverse_kinematics(x, y, phi, l1, l2, l3):
    """
    Calcula las soluciones de cinemática inversa para un robot 3R en el plano.
    x, y: posición deseada del efector final.
    phi: orientación deseada del efector final (en radianes).
    l1, l2, l3: longitudes de los tres eslabones.
    
    Retorna dos tuplas con los ángulos (θ1, θ2, θ3) correspondientes a las dos configuraciones (si existen).
    """
    # Posición del "wrist" (punto de unión entre el segundo y tercer eslabón)
    wx = x - l3 * np.cos(phi)
    wy = y - l3 * np.sin(phi)
    
    # Resolver la parte de dos eslabones para alcanzar (wx, wy)
    cos_theta2 = (wx**2 + wy**2 - l1**2 - l2**2) / (2 * l1 * l2)
    # Verificar alcanzabilidad
    if abs(cos_theta2) > 1:
        return []  # No es alcanzable
    sin_theta2 = np.sqrt(1 - cos_theta2**2)
    
    # Dos posibles soluciones para θ₂
    theta2_1 = np.arctan2(sin_theta2, cos_theta2)
    theta2_2 = np.arctan2(-sin_theta2, cos_theta2)
    
    # Calcular θ₁ para cada solución
    k1 = l1 + l2 * cos_theta2
    k2_1 = l2 * sin_theta2
    k2_2 = -l2 * sin_theta2
    
    theta1_1 = np.arctan2(wy, wx) - np.arctan2(k2_1, k1)
    theta1_2 = np.arctan2(wy, wx) - np.arctan2(k2_2, k1)
    
    # Calcular θ₃ para cada configuración
    theta3_1 = phi - (theta1_1 + theta2_1)
    theta3_2 = phi - (theta1_2 + theta2_2)
    
    return (theta1_1, theta2_1, theta3_1), (theta1_2, theta2_2, theta3_2)

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

class Goal:
    def __init__(self, x, y, phi=0):
        self._x = x
        self._y = y
        self._phi = phi  # Agregamos la orientación
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

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        if self._phi != value:
            self._phi = value
            self._cspace = None

    def get_pspace_pos(self):
        # Ahora se retorna la posición junto con la orientación
        return self._x, self._y, self._phi

    def get_cspace_pos(self, arm, wrap=True, as_degrees=True):
        if self._cspace is None:
            if arm.num_joints == 2:
                # Para un brazo de dos eslabones, se ignora phi (o se asume una solución particular)
                self._cspace = inverse_kinematics(self._x, self._y, arm.dh[0]['r'], arm.dh[1]['r'])
            elif arm.num_joints == 3:
                # Para un brazo de tres eslabones, se utiliza phi en la cinemática inversa.
                # Se asume que tienes definida una función inverse_kinematics que reciba (x, y, phi, l1, l2, l3)
                self._cspace = inverse_kinematics(self._x, self._y, self._phi,
                                                  arm.dh[0]['r'], arm.dh[1]['r'], arm.dh[2]['r'])
            else:
                raise ValueError("Configuración de brazo no soportada")
            if as_degrees:
                self._cspace = [tuple(np.degrees(theta) for theta in solution) for solution in self._cspace]
            if wrap:
                self._cspace = [tuple(theta % 360 for theta in solution) for solution in self._cspace]
        return tuple(self._cspace)
    
    def get_paths(self, arm):
        # Igual que antes, se generan las rutas a partir del c-space
        current_angles = arm.get_cspace_pos(wrap=True, as_degrees=True)
        wrapped_goals = self.get_cspace_pos(arm)

        candidates = []
        for candidate in wrapped_goals:
            virtual_candidates = quadrant.compute_all_quadrants(candidate, 360)
            for candidate_shifted in virtual_candidates:
                diff0 = candidate_shifted[0] - current_angles[0]
                diff1 = candidate_shifted[1] - current_angles[1]
                diff2 = candidate_shifted[2] - current_angles[2]
                dist = np.linalg.norm([diff0, diff1, diff2])
                candidates.append((dist, candidate_shifted))
        candidates.sort(key=lambda x: x[0])
        return candidates

    def get_path(self, arm, goal_index=0):
        candidates = self.get_paths(arm)
        return candidates[goal_index % len(candidates)][1]

    def pathfind(self, arm, collisions):
        candidates = self.get_paths(arm)
        for _, candidate in candidates:
            if not Goal.path_collides(arm.get_cspace_pos(), candidate, collisions):
                return candidate

    @staticmethod
    def path_collides(start_cspace, target_cspace, collision_list, threshold=10.0):
        p_start = np.array(start_cspace)
        p_target = np.array(target_cspace)
        if quadrant.get_quadrant(target_cspace, 360) != (0, 0, 0):
            target_quad = quadrant.get_quadrant(target_cspace, 360)
            opposite_quad = quadrant.get_opposite_quadrant(target_quad)
            p_start = np.array(quadrant.compute_quadrant(start_cspace, opposite_quad, 360))
            p_target = np.array(quadrant.compute_quadrant(target_cspace, (0, 0, 0), 360))
        
        for collision in collision_list:
            collision_np = np.array(collision)
            dist = line_segment_distance(p_start, p_target, collision_np)
            if dist < threshold:
                return True
        return False

class Arm:
    def __init__(self, dh=None):
        self.dh = dh or [
            {'theta': np.radians(0), 'r': 1},
            {'theta': np.radians(0), 'r': 1},
            {'theta': np.radians(0), 'r': 1},
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
            x = last_x + link['r'] * np.cos(theta)
            y = last_y + link['r'] * np.sin(theta)
            positions.append((x, y))
        return positions
