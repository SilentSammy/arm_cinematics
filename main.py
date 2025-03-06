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
arm = Arm([
            {'theta': math.radians(0), 'r': 1},
            {'theta': math.radians(0), 'r': 1},
            # {'theta': math.radians(0), 'r': 1},
        ])
obstacle = (1, 1, 0.5) # x, y, radius
goal = Goal(-0.5, 0.75)  # goal position in physical space
target = None
cspace_collisions = []

# Initialize plotting
plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
pspace_drawer = PSpaceDrawer(ax[0], arm, goal, obstacle)
cspace_drawer = CSpaceDrawer(fig, arm, goal, obstacle)

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
        if len(arm.dh) > 2:
            arm.dh[2]['theta'] += np.radians(45) * dt

        # Redraw the canvas
        draw()

def move_to_goal():
    global goal_index
    goal_index = 0
    cspace_drawer.closest_goal = None # Avoids the line drawing bug

    # Get the starting position (in degrees)
    start = arm.get_cspace_pos(wrap=True, as_degrees=True)

    # Define animation parameters:
    anim_progress = 0.0   # progress from 0.0 (start) to 1.0 (goal)
    anim_speed = 0.25      # speed factor (adjust as needed)
    
    dt = 0
    last_time = None
    while anim_progress < 1.0:
        dt = time.time() - last_time if last_time else 0
        last_time = time.time()
        
        # Update animation progress
        anim_progress = min(anim_progress + dt * anim_speed, 1.0)

        # Compute new angles by linear interpolation
        new_angles = [ start[i] + anim_progress * (target[i] - start[i]) for i in range(len(start)) ]
        arm.dh[0]['theta'] = np.radians(new_angles[0])
        arm.dh[1]['theta'] = np.radians(new_angles[1])
        if arm.num_joints > 2:
            arm.dh[2]['theta'] = np.radians(new_angles[2])
    
        draw()
    print("Goal reached!")

def teleport_to_goals():
    ik_sols = goal.get_cspace_pos(arm)

    if False:
        for i, angle in enumerate(ik_sols[2]):
            arm.dh[i]['theta'] = np.radians(angle)
    else:
        for sol in ik_sols:
            for i, angle in enumerate(sol):
                arm.dh[i]['theta'] = np.radians(angle)
            draw()
        
def scan(animate=False):
    global cspace_collisions
    cspace_collisions.clear()
    for c in arm.cspace_scanner(obstacle):
        if c is not None:
            cspace_collisions.extend(c)
        elif animate:
            cspace_drawer.draw_collisions(cspace_collisions)
            draw()

def pathfind():
    # goals = goal.get_cspace_pos(arm)
    global target
    closest_goal = goal.pathfind(arm, cspace_collisions)
    target = cspace_drawer.closest_goal = closest_goal

def control():
    global target
    a_vel = np.radians(90)  # Angular velocity in radians per second
    lin_vel = 1  # Linear velocity in units per second
    control_mode = 1  # Default control mode (1 for arm, 2 for goal)

    dt = 0
    last_time = None
    while True:
        dt = time.time() - last_time if last_time else 0
        last_time = time.time()

        # Change control mode (1 for arm, 2 for goal)
        if is_key_down('1'):
            control_mode = 1
            print("Control mode: Arm")
        elif is_key_down('2'):
            control_mode = 2
            print("Control mode: Goal")
        elif is_key_down('3'):
            print("Control mode: Scan")
            scan(True)
        elif is_key_down('4'):
            print("Teleporting to goals")
            teleport_to_goals()
        elif is_key_down('6'):
            print("Moving to goal")
            move_to_goal()
        elif is_key_down('0'):
            cspace_drawer.draw_collisions(cspace_collisions)
            print("Showing collisions")
        elif not is_key_down('5'):
            cspace_drawer.draw_collisions([])

        # Control the arm or goal based on the control mode
        if control_mode == 1:
            arm.dh[0]['theta'] += a_vel * dt * (1 if is_key_down('right') else -1 if is_key_down('left') else 0)
            arm.dh[1]['theta'] += a_vel * dt * (1 if is_key_down('up') else -1 if is_key_down('down') else 0)
            if len(arm.dh) > 2:
                arm.dh[2]['theta'] += a_vel * dt * (1 if is_key_down('d') else -1 if is_key_down('a') else 0)
            if any(is_key_down(key) for key in ['up', 'down', 'left', 'right']):
                # print(f"Joint 1: {np.degrees(arm.dh[0]['theta']):.2f}°, Joint 2: {np.degrees(arm.dh[1]['theta']):.2f}°")
                pass
        elif control_mode == 2:
            goal.y += lin_vel * dt * (1 if is_key_down('up') else -1 if is_key_down('down') else 0)
            goal.x += lin_vel * dt * (1 if is_key_down('right') else -1 if is_key_down('left') else 0)
            if any(is_key_down(key) for key in ['up', 'down', 'left', 'right']):
                print(f"Goal Position: ({goal.x:.2f}, {goal.y:.2f})")
        
        # Pathfind if any movement keys are pressed
        if any(is_key_down(key) for key in ['up', 'down', 'left', 'right', 'a', 'd']):
            pathfind()
        
        # Redraw
        draw()
            
if __name__ == '__main__':
    scan(False)
    draw()
    control()
