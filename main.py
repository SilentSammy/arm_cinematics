import math
import time
import numpy as np
import matplotlib.pyplot as plt
from arm import Arm, Goal
from drawer import PSpaceDrawer, CSpaceDrawer

def listen_for_keys(fig):
    pressed_keys = set()
    pressed_once_keys = set()
    new_press_keys = set()

    def on_key_press(event):
        if event.key not in pressed_keys:
            new_press_keys.add(event.key)
        pressed_keys.add(event.key)

    def on_key_release(event):
        pressed_keys.discard(event.key)
        pressed_once_keys.discard(event.key)
        new_press_keys.discard(event.key)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    def is_key_down(key):
        return key in pressed_keys

    def is_key_pressed(key):
        if key in new_press_keys:
            new_press_keys.discard(key)
            pressed_once_keys.add(key)
            return True
        return False

    return is_key_down, is_key_pressed

# Objects to plot
arm = Arm()
obstacle = (1, 1, 0.4) # x, y, radius
goal = Goal(-0.5, 0.75)  # goal position in physical space
goal_index = 0
target = None

# Initialize plotting
plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax1 = ax[0]
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

pspace_drawer = PSpaceDrawer(ax1, arm, goal, obstacle)
cspace_drawer = CSpaceDrawer(ax2, arm, goal, obstacle)

# Listen for key presses
is_key_down, key_pressed = listen_for_keys(fig)

def draw():
    pspace_drawer.draw()
    cspace_drawer.draw()
    fig.canvas.flush_events()

def spin():
    dt = 0
    last_time = None
    while True:
        dt = time.time() - last_time if last_time else 0
        last_time = time.time()

        arm.dh[0]['theta'] += np.radians(180) * dt
        arm.dh[1]['theta'] += np.radians(90) * dt
        
        # Redraw the canvas
        draw()

def pathfind():
    global goal_index
    goal_index = 0
    cspace_drawer.closest_goal = None  # Evita errores en la línea de conexión

    # Obtener la posición inicial (en grados)
    start = arm.get_cspace_pos(wrap=True, as_degrees=True)

    # Calcular la solución IK (camino) a partir del objetivo
    target = goal.pathfind(arm, cspace_drawer.collisions)
    if target is None:
        print("No se encontró un camino libre de colisión.")
        return  # O maneja el error de otra forma

    # Parámetros de animación
    anim_progress = 0.0  # progreso de 0.0 (inicio) a 1.0 (meta)
    anim_speed = 0.5     # factor de velocidad (ajustable)

    dt = 0
    last_time = None
    while anim_progress < 1.0:
        dt = time.time() - last_time if last_time else 0
        last_time = time.time()

        # Actualizar el progreso de la animación
        anim_progress = min(anim_progress + dt * anim_speed, 1.0)

        # Interpolación lineal entre la posición inicial y el objetivo
        new_angles = [start[i] + anim_progress * (target[i] - start[i])
                      for i in range(len(start))]
        arm.dh[0]['theta'] = np.radians(new_angles[0])
        arm.dh[1]['theta'] = np.radians(new_angles[1])
        arm.dh[2]['theta'] = np.radians(new_angles[2])

        draw()
    print("¡Objetivo alcanzado!")

def control():
    global goal_index, target
    a_vel = np.radians(90)  # Angular velocity in radians per second
    lin_vel = 1  # Linear velocity in units per second
    control_mode = 1  # Default control mode (1 for arm, 2 for goal)

    dt = 0
    last_time = None
    while True:
        dt = time.time() - last_time if last_time else 0
        last_time = time.time()

        # Choose the next goal
        d_pressed, a_pressed = key_pressed('d'), key_pressed('a')
        goal_index += 1 if d_pressed else -1 if a_pressed else 0
        if d_pressed or a_pressed:
            print(f"Goal {goal_index}: {cspace_drawer.closest_goal}")

        # Change control mode (1 for arm, 2 for goal)
        if is_key_down('1'):
            control_mode = 1
            print("Control mode: Arm")
        elif is_key_down('2'):
            control_mode = 2
            print("Control mode: Goal")
        elif is_key_down('p'):
            print("Control mode: Pathfinding")
            pathfind()

        # Control the arm or goal based on the control mode
        if control_mode == 1:
            arm.dh[1]['theta'] += a_vel * dt * (1 if is_key_down('up') else -1 if is_key_down('down') else 0)
            arm.dh[0]['theta'] += a_vel * dt * (1 if is_key_down('right') else -1 if is_key_down('left') else 0)
            

            if any(is_key_down(key) for key in ['up', 'down', 'left', 'right']):
                print(f"Joint 1: {np.degrees(arm.dh[0]['theta']):.2f}°, Joint 2: {np.degrees(arm.dh[1]['theta']):.2f}°")
        elif control_mode == 2:
            goal.y += lin_vel * dt * (1 if is_key_down('up') else -1 if is_key_down('down') else 0)
            goal.x += lin_vel * dt * (1 if is_key_down('right') else -1 if is_key_down('left') else 0)
            if any(is_key_down(key) for key in ['up', 'down', 'left', 'right']):
                print(f"Goal Position: ({goal.x:.2f}, {goal.y:.2f})")
        
        # Only redraw if a key was pressed
        if any(is_key_down(key) for key in ['up', 'down', 'left', 'right', '1', '2']) or d_pressed or a_pressed:
            cspace_drawer.closest_goal = target = goal.pathfind(arm, cspace_drawer.collisions)
            draw()
        else: fig.canvas.flush_events()
            
if __name__ == '__main__':
    cspace_drawer.scan()
    draw()
    control()
