import math
import time
import numpy as np
import matplotlib.pyplot as plt
from arm import Arm, Goal
from drawer import PSpaceDrawer, CSpaceDrawer

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
    key_state = {'up': False, 'down': False, 'left': False, 'right': False}
    control_mode = 1  # Default control mode (1 for arm, 2 for goal)

    def on_key_press(event):
        nonlocal control_mode
        if event.key in key_state:
            key_state[event.key] = True
        elif event.key == '1':
            control_mode = 1
            print("Control mode: Arm")
        elif event.key == '2':
            control_mode = 2
            print("Control mode: Goal")

    def on_key_release(event):
        if event.key in key_state:
            key_state[event.key] = False

    # Connect the key press and release events to the callbacks
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)
    dt = 0
    last_time = None
    while True:
        dt = time.time() - last_time if last_time else 0
        last_time = time.time()

        if control_mode == 1:
            if key_state['up']:
                arm.dh[1]['theta'] += a_vel * dt
            if key_state['down']:
                arm.dh[1]['theta'] -= a_vel * dt
            if key_state['left']:
                arm.dh[0]['theta'] -= a_vel * dt
            if key_state['right']:
                arm.dh[0]['theta'] += a_vel * dt
            if any(key_state.values()):
                print(f"Joint 1: {np.degrees(arm.dh[0]['theta']):.2f}°, Joint 2: {np.degrees(arm.dh[1]['theta']):.2f}°")
        elif control_mode == 2:
            if key_state['up']:
                goal.y += lin_vel * dt
            if key_state['down']:
                goal.y -= lin_vel * dt
            if key_state['left']:
                goal.x -= lin_vel * dt
            if key_state['right']:
                goal.x += lin_vel * dt
            if any(key_state.values()):
                print(f"Joint 1: {np.degrees(arm.dh[0]['theta']):.2f}°, Joint 2: {np.degrees(arm.dh[1]['theta']):.2f}°")

        draw()

if __name__ == '__main__':
    scan_gen = cspace_drawer.scan_generator()
    for _ in scan_gen:
        cspace_drawer.draw_collisions()
        draw()
    control_arm()
