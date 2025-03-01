import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
        
class Arm:
    def __init__(self, dh=None):
        self.dh = dh or [
            {'theta': np.radians(0), 'r': 2},
            # {'theta': np.radians(0), 'r': 2},
        ]

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

class ArmDrawer:
    def __init__(self, arm, ax):
        self.arm = arm
        self.ax = ax
        self.point_artists = []  # to hold the plotted joint points
        self.line_artists = []   # to hold the connecting line segments

    def draw(self):
        """
        Redraw the arm based on the Arm's current joint positions.
        """
        positions = self.arm.get_joint_positions()
        
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
    
    def get_artists(self):
        return self.point_artists + self.line_artists
                

# Cinematic variables
dt = 0
start_time = None
last_time = None

# Create a figure with two subplots
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Initialize empty data for both plots
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
z = np.linspace(0, 2 * np.pi, 100)

# Objects
arm = Arm()
arm_drawer = ArmDrawer(arm, ax1)
obstacle = (1, 1, 0.25) # x, y, radius

# Set titles and labels
ax1.set_title('Physical Space')
ax2.set_title('C-Space')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# Set equal aspect ratio and centered axes for the 2D plot
ax1.set_aspect('equal', 'box')
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)

# Draw obstacle
ax1.add_patch(plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', alpha=0.5))

def line_segment_distance(p1, p2, p):
    """Calcula la distancia mínima entre el punto p y el segmento definido por p1 -> p2."""
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

def scan():
    i = 0
    while i < 360:
        arm.dh[0]['theta'] = np.radians(i)
        pos = arm.get_joint_positions()

        # Get the distance between the end effector and the center of the obstacle
        dist = line_segment_distance(np.array(pos[0]), np.array(pos[1]), np.array(obstacle[:2]))

        print(f"Angle: {i}, Distance: {dist}", "Collision!" if dist < obstacle[2] else "")

        # Adjust the increment based on the distance to the obstacle
        increment = min(max(3, int(dist * 15)), 15)  # Example: proportional to distance
        i += increment

# Function to update the point position
def update_sim(frame):
    global dt, last_time
    dt = time.time() - last_time if last_time else 0
    last_time = time.time()

    # Update the arm points and lines
    arm.dh[0]['theta'] += dt * np.pi / 20
    # arm.dh[1]['theta'] += dt * np.pi / 20
    arm_drawer.draw()

    # Return both points and lines so they get redrawn
    return arm_drawer.get_artists()

def start_animation():
    # Create an animation
    ani = animation.FuncAnimation(fig, update_sim, frames=len(x), interval=10, blit=True)

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    scan()