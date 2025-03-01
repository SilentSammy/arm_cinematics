import time
import matplotlib.pyplot as plt
import math

# Initialize plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# First subplot
point1, = ax1.plot([], [], 'o', label='Moving Point 1')
line1, = ax1.plot([], [], label='Line to Point 1')
ax1.set_title("Animated Point 1 Moving in a Circle")
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()

# Second subplot
point2, = ax2.plot([], [], 'o', label='Moving Point 2')
line2, = ax2.plot([], [], label='Line to Point 2')
ax2.set_title("Animated Point 2 Moving in a Circle")
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.legend()

start_time = time.time()

x = 0
while True:
    # Calculate the position of the first point
    x_pos1 = math.cos(x)
    y_pos1 = math.sin(x)
    
    # Calculate the position of the second point (phase shifted)
    x_pos2 = math.cos(x + math.pi / 2)
    y_pos2 = math.sin(x + math.pi / 2)
    
    # Update the first point's position
    point1.set_data([x_pos1], [y_pos1])
    
    # Update the first line's position
    line1.set_data([0, x_pos1], [0, y_pos1])
    
    # Update the second point's position
    point2.set_data([x_pos2], [y_pos2])
    
    # Update the second line's position
    line2.set_data([0, x_pos2], [0, y_pos2])
    
    # Redraw the canvas
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    
    x += 0.1
