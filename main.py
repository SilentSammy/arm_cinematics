import math
import time
import numpy as np
import matplotlib.pyplot as plt
from arm import Arm, Goal
from drawer import PSpaceDrawer, CSpaceDrawer

def listen_for_keys(fig):
    pressed_keys = set()
    def on_key_press(event):
        pressed_keys.add(event.key)

    def on_key_release(event):
        pressed_keys.discard(event.key)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    def is_key_pressed(key):
        return key in pressed_keys

    return is_key_pressed

def draw():
    pspace_drawer.draw()
    cspace_drawer.draw()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

# Objects to plot
arm = Arm()
obstacle = (1, 1, 0.25) # x, y, radius
goal = Goal(-0.5, 0.75)  # goal position in physical space

# Initialize plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
pspace_drawer = PSpaceDrawer(ax1, arm, goal, obstacle)
cspace_drawer = CSpaceDrawer(ax2, arm, goal, obstacle)

is_key_pressed = listen_for_keys(fig)

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

def control_arm():
    a_vel = np.radians(90)  # Angular velocity in radians per second
    lin_vel = 1  # Linear velocity in units per second
    control_mode = 1  # Default control mode (1 for arm, 2 for goal)

    dt = 0
    last_time = None
    while True:
        dt = time.time() - last_time if last_time else 0
        last_time = time.time()

        if is_key_pressed('1'):
            control_mode = 1
            print("Control mode: Arm")
        elif is_key_pressed('2'):
            control_mode = 2
            print("Control mode: Goal")

        if control_mode == 1:
            if is_key_pressed('up'):
                arm.dh[1]['theta'] += a_vel * dt
            if is_key_pressed('down'):
                arm.dh[1]['theta'] -= a_vel * dt
            if is_key_pressed('left'):
                arm.dh[0]['theta'] -= a_vel * dt
            if is_key_pressed('right'):
                arm.dh[0]['theta'] += a_vel * dt
            if any(is_key_pressed(key) for key in ['up', 'down', 'left', 'right']):
                print(f"Joint 1: {np.degrees(arm.dh[0]['theta']):.2f}°, Joint 2: {np.degrees(arm.dh[1]['theta']):.2f}°")
        elif control_mode == 2:
            if is_key_pressed('up'):
                goal.y += lin_vel * dt
            if is_key_pressed('down'):
                goal.y -= lin_vel * dt
            if is_key_pressed('left'):
                goal.x -= lin_vel * dt
            if is_key_pressed('right'):
                goal.x += lin_vel * dt
            if any(is_key_pressed(key) for key in ['up', 'down', 'left', 'right']):
                print(f"Goal Position: ({goal.x:.2f}, {goal.y:.2f})")
        draw()

if __name__ == '__main__':
    scan_gen = cspace_drawer.scan_generator()
    for point in scan_gen:
        pass
    draw()
    control_arm()
