import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# ----------------------------
# Parámetros del robot y obstáculo
# ----------------------------
L1 = 2.0
L2 = 2.0
L3 = 2.0

# Obstáculo puntual en el espacio de trabajo
cx, cy = 1.2, 1.2  # Coordenadas del obstáculo
tolerance = 0.01   # Umbral para considerar colisión

# ----------------------------
# Funciones básicas
# ----------------------------
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

def check_collision(theta1, theta2, theta3):
    """Devuelve True si el brazo en (theta1, theta2, theta3) colisiona con el obstáculo puntual."""
    p0 = np.array([0.0, 0.0])
    p1 = p0 + L1 * np.array([np.cos(theta1), np.sin(theta1)])
    p2 = p1 + L2 * np.array([np.cos(theta1 + theta2), np.sin(theta1 + theta2)])
    p3 = p2 + L3 * np.array([np.cos(theta1 + theta2 + theta3), np.sin(theta1 + theta2 + theta3)])
    p_obs = np.array([cx, cy])
    
    dist1 = line_segment_distance(p0, p1, p_obs)
    dist2 = line_segment_distance(p1, p2, p_obs)
    dist3 = line_segment_distance(p2, p3, p_obs)
    
    return (dist1 <= tolerance) or (dist2 <= tolerance) or (dist3 <= tolerance)

def forward_kinematics(theta1, theta2, theta3):
    """Calcula las posiciones (p0, p1, p2, p3) en el espacio de trabajo."""
    p0 = np.array([0.0, 0.0])
    p1 = p0 + L1 * np.array([np.cos(theta1), np.sin(theta1)])
    p2 = p1 + L2 * np.array([np.cos(theta1+theta2), np.sin(theta1+theta2)])
    p3 = p2 + L3 * np.array([np.cos(theta1+theta2+theta3), np.sin(theta1+theta2+theta3)])
    return p0, p1, p2, p3

def inverse_kinematics(x, y, phi):
    """
    Calcula las soluciones de cinemática inversa para un robot planar 3R.
    x, y: posición deseada del efector final.
    phi: orientación deseada del efector final.
    Retorna una lista con las posibles soluciones [theta1, theta2, theta3].
    Si no es alcanzable, retorna una lista vacía.
    """
    # Calcular la posición del "wrist" (centro del tercer eslabón)
    wx = x - L3 * np.cos(phi)
    wy = y - L3 * np.sin(phi)
    d = np.sqrt(wx**2 + wy**2)
    # Comprobar alcanzabilidad para el brazo 2R (primeros dos eslabones)
    if d > (L1 + L2) or d < abs(L1 - L2):
        return []
    # Ley de cosenos para theta2
    cos_theta2 = (wx**2 + wy**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if cos_theta2 < -1 or cos_theta2 > 1:
        return []
    theta2_sol1 = np.arccos(cos_theta2)      # Solución "codo abajo"
    theta2_sol2 = -np.arccos(cos_theta2)     # Solución "codo arriba"
    
    solutions = []
    for theta2 in [theta2_sol1, theta2_sol2]:
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        theta1 = np.arctan2(wy, wx) - np.arctan2(k2, k1)
        theta3 = phi - (theta1 + theta2)
        solutions.append([theta1, theta2, theta3])
    return solutions

# ----------------------------
# Precomputación de la región de colisión en el C-space
# ----------------------------
N = 20  # Número de muestras por ángulo
theta1_values = np.linspace(0, 2*np.pi, N)
theta2_values = np.linspace(0, 2*np.pi, N)
theta3_values = np.linspace(0, 2*np.pi, N)

coll_pts = []  # Lista para puntos de colisión en el C-space

for i, t1 in enumerate(theta1_values):
    for j, t2 in enumerate(theta2_values):
        for k, t3 in enumerate(theta3_values):
            if check_collision(t1, t2, t3):
                coll_pts.append((t1, t2, t3))
                
if coll_pts:
    coll_pts = np.array(coll_pts)
    coll_t1 = coll_pts[:,0]
    coll_t2 = coll_pts[:,1]
    coll_t3 = coll_pts[:,2]
else:
    coll_t1 = coll_t2 = coll_t3 = np.array([])

# ----------------------------
# Variables globales para la simulación interactiva
# ----------------------------
angles = [0.0, 0.0, 0.0]      # Configuración actual del robot (en radianes)
# Ahora, en lugar de goal_angles, definimos el objetivo en el espacio de trabajo:
goal_pose = [3.0, 0.0, 0.0]   # [x, y, phi] deseados para el efector final
# La solución de IK (goal_angles) se calculará cuando se presione 'p'

# ----------------------------
# Configuración de la figura y ejes
# ----------------------------
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1,2,1)              # Espacio de trabajo (2D)
ax2 = fig.add_subplot(1,2,2, projection='3d')  # Espacio de configuraciones (C-space)

def update_plot():
    ax1.clear()
    ax2.clear()
    
    # Cinemática directa
    theta1, theta2, theta3 = angles
    p0, p1, p2, p3 = forward_kinematics(theta1, theta2, theta3)
    
    # --- Espacio de Trabajo (ax1) ---
    ax1.plot([p0[0], p1[0]], [p0[1], p1[1]], 'b-', lw=3)
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=3)
    ax1.plot([p2[0], p3[0]], [p2[1], p3[1]], 'b-', lw=3)
    ax1.scatter([p0[0], p1[0], p2[0], p3[0]], [p0[1], p1[1], p2[1], p3[1]], c='k', zorder=5)
    ax1.scatter(cx, cy, c='r', marker='x', s=100, label='Obstáculo')
    goal_x, goal_y, goal_phi = goal_pose
    ax1.scatter(goal_x, goal_y, c='green', s=100, label='Objetivo')
    ax1.arrow(goal_x, goal_y, 0.5*np.cos(goal_phi), 0.5*np.sin(goal_phi),
              head_width=0.1, head_length=0.1, fc='green', ec='green')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.set_title("Espacio de Trabajo")
    ax1.legend()
    
    # --- Espacio de Configuraciones (ax2) ---
    # Mostrar la región de colisión precomputada
    if coll_t1.size > 0:
        ax2.scatter(coll_t1, coll_t2, coll_t3, c='red', s=10, label='Región de Colisión')
    
    # Calcular la solución IK para el objetivo en el C-space
    ik_solutions = inverse_kinematics(goal_x, goal_y, goal_phi)
    if ik_solutions and len(ik_solutions) > 0:
        # Se elige, por ejemplo, la primera solución
        goal_conf = ik_solutions[0]
        ax2.scatter(goal_conf[0], goal_conf[1], goal_conf[2], c='green', s=100, label='Conf. Objetivo')
        ax2.plot([theta1, goal_conf[0]], [theta2, goal_conf[1]], [theta3, goal_conf[2]],
                 'k--', label='Trayectoria Deseada')
    else:
        print("No se encontró solución IK para el objetivo.")
    
    ax2.scatter(theta1, theta2, theta3, c='purple', s=100, label='Configuración Actual')
    ax2.set_xlabel(r'$\theta_1$')
    ax2.set_ylabel(r'$\theta_2$')
    ax2.set_zlabel(r'$\theta_3$')
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-np.pi, np.pi)
    ax2.set_zlim(-np.pi, np.pi)
    ax2.set_title('Espacio de Configuraciones')
    ax2.legend()
    
    fig.canvas.draw()

def ease_in_out(t):
    """Función de suavizado: acelera al principio y desacelera al final."""
    return 3*t**2 - 2*t**3

def plan_path(start, goal, num_steps=100):
    """
    Interpola entre la configuración inicial y la meta usando una interpolación no lineal (ease in/out).
    Si algún punto intermedio colisiona, se cancela la trayectoria.
    """
    path = []
    for i in range(num_steps+1):
        t = i / num_steps
        tau = ease_in_out(t)  # Usa el suavizado en lugar de t lineal
        config = (1 - tau) * np.array(start) + tau * np.array(goal)
        if check_collision(config[0], config[1], config[2]):
            print("Colisión en paso", i, "con config =", config)
            return None
        path.append(config)
    return path

def plan_path_curve(start, goal, control, num_steps=100):
    """
    Genera una trayectoria curva (Bézier cuadrática) en el espacio de configuraciones.
    start: configuración inicial [theta1, theta2, theta3]
    goal: configuración final [theta1, theta2, theta3]
    control: punto de control en el C-space para definir la curva
    num_steps: número de pasos en la interpolación
    Retorna una lista de configuraciones a lo largo de la trayectoria o None si se detecta colisión.
    """
    path = []
    for i in range(num_steps+1):
        t = i / num_steps
        config = (1 - t)**2 * np.array(start) + 2*(1-t)*t * np.array(control) + t**2 * np.array(goal)
        if check_collision(config[0], config[1], config[2]):
            print("Colisión en paso", i, "con config =", config)
            return None
        path.append(config)
    return path

def animate_path(path, delay=0.1):
    """Anima el movimiento actualizando la configuración a lo largo de la trayectoria."""
    global angles
    for config in path:
        angles = list(config)
        update_plot()
        plt.pause(delay)

def on_key(event):
    """
    Maneja los eventos de teclado:
    - w/s: Incrementa/disminuye la coordenada x del objetivo.
    - d/a: Incrementa/disminuye la coordenada y del objetivo.
    - q/e: Incrementa/disminuye la orientación φ del objetivo.
    - p: Calcula la solución IK para el objetivo, planifica y anima la trayectoria.
    """
    global goal_pose, angles
    delta = 0.1
    if event.key == 'up':
        goal_pose[1] += delta
    elif event.key == 'down':
        goal_pose[1] -= delta
    elif event.key == 'd':
        goal_pose[0] += delta
    elif event.key == 'a':
        goal_pose[0] -= delta
    elif event.key == '1':
        goal_pose[2] += delta
    elif event.key == '2':
        goal_pose[2] -= delta
    elif event.key == 'p':
        print("Planificando trayectoria hacia el objetivo (x={:.2f}, y={:.2f}, φ={:.2f})...".format(
            goal_pose[0], goal_pose[1], goal_pose[2]))
        # Obtener todas las soluciones IK para el objetivo
        ik_solutions = inverse_kinematics(goal_pose[0], goal_pose[1], goal_pose[2])
        if not ik_solutions:
            print("El objetivo no es alcanzable.")
        else:
            # Filtrar las soluciones que sean libres de colisión
            free_solutions = [sol for sol in ik_solutions if not check_collision(sol[0], sol[1], sol[2])]
            if not free_solutions:
                print("No existe solución IK libre de colisión para ese objetivo.")
            else:
                # Por ejemplo, se elige la primera solución libre
                goal_angles = free_solutions[0]
                print("Solución IK seleccionada:", goal_angles)
                path = plan_path(angles, goal_angles)
                if path is not None:
                    animate_path(path)
                else:
                    print("La trayectoria directa presenta colisión.")

    update_plot()

# Conectar la función de manejo de teclas a la figura
fig.canvas.mpl_connect('key_press_event', on_key)

# Dibujar la configuración inicial
update_plot()
print("Controles:")
print("  w/s: Incrementa/disminuye x del objetivo")
print("  d/a: Incrementa/disminuye y del objetivo")
print("  q/e: Incrementa/disminuye φ del objetivo")
print("  p: Planificar y animar trayectoria hacia el objetivo")

plt.show()