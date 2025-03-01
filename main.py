import math
import time
import numpy as np
import matplotlib.pyplot as plt
from arm import Arm, ArmDrawer, Goal

class PSpaceDrawer:
    pass

class CSpaceDrawer:
    pass

def draw():
    global arm_pos, arm_drawer
    angles = arm.get_joint_angles()
    
    # draw the arm in physical space
    arm_drawer.draw()

    # draw the current arm position in C-space
    arm_pos[0].set_data([angles[0]], [angles[1]])

    # plot the goal position in physical space
    goal_point[0].set_data([goal.x], [goal.y])
    c_space_points = goal.get_pos_cspace(arm)
    for i in range(len(c_space_points)):
        goal_point_c[i].set_data([c_space_points[i][0]], [c_space_points[i][1]])

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

# Initialize plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First subplot: Physical Space
ax1.set_title('Physical Space')
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

# Second subplot: C-space
ax2.set_title('C-space')
ax2.set_xlim(0, 360)
ax2.set_ylim(0, 360)
ax2.set_xlabel("Joint 1 Angle (degrees)")
ax2.set_ylabel("Joint 2 Angle (degrees)")

# Objects to plot
arm = Arm()
arm_drawer = ArmDrawer(arm, ax1)
arm_pos = ax2.plot(0, 0, 'bo') # plot the current position in C-space
obstacle = (1, 1, 0.25) # x, y, radius
ax1.add_patch(plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', alpha=0.5)) # draw just once
goal = Goal(-0.5, 0.75)  # goal position in physical space
goal_point = ax1.plot([goal.x], [goal.y], 'go')  # plot the goal position in physical space
goal_point_c = []
for p in goal.get_pos_cspace(arm):
    goal_point_c.append(ax2.plot(p[0], p[1], 'go')[0])  # plot the goal position in C-space

# List to store collision points in C-space
c_space_points = []

def scan(visualize=True):
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
    
    i = 0
    while i < 360:
        j = 0
        closest_dist = None
        while j < 360:
            arm.dh[0]['theta'] = np.radians(i)
            arm.dh[1]['theta'] = np.radians(j)
            pos = arm.get_joint_positions()

            # Get distances for the two segments
            dist1 = line_segment_distance(np.array(pos[0]), np.array(pos[1]), np.array(obstacle[:2]))
            dist2 = line_segment_distance(np.array(pos[1]), np.array(pos[2]), np.array(obstacle[:2]))
            dist = min(dist1, dist2)
            closest_dist = dist if closest_dist is None else min(closest_dist, dist)

            # Determine collision status
            collision1 = dist1 < obstacle[2]
            collision2 = dist2 < obstacle[2]
            collision = collision1 or collision2
            print(f"Joint 1 Angle: {i}, Joint 2 Angle: {j}, Distance 1: {dist1}, Distance 2: {dist2}", 
                  "Collision!" if collision else "")
            
            # Store collision points and update current C-space position
            if collision:
                c_space_points.append((i, j))
                # Bulk-add points if first segment collides
                if collision1:
                    for k in range(j, min(j + 360, 361), 20):
                        c_space_points.append((i, k))
                        if visualize:
                            ax2.plot(i, k, 'ro')
            if visualize:
                # Update current C-space position marker
                arm_pos[0].set_data([i], [j])
                # Draw collision points so far
                if c_space_points:
                    x_data, y_data = zip(*c_space_points)
                    ax2.plot(x_data, y_data, 'ro')
                # Update arm drawing and canvas
                arm_drawer.draw()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            # Adjust increments; if first segment collides, skip rest of j-values in this iteration
            increment1 = min(max(5, int(closest_dist * 20)), 90)  # first joint increment remains unchanged
            increment2 = min(max(10, int(dist2 * 30)), 90)
            if collision1:
                increment2 = 360  # Skip checking second segment details for this i value
            j += increment2
        i += increment1

    # Final bulk draw for collision points if not visualizing in real time
    if not visualize and c_space_points:
        x_data, y_data = zip(*c_space_points)
        ax2.plot(x_data, y_data, 'ro')
        fig.canvas.draw_idle()
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
            print(f"Goal Position: x={goal.x:.2f}, y={goal.y:.2f}")

        draw()

if __name__ == '__main__':
    scan(False)
    control_arm()
