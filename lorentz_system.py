import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

def lorenz_system(t, state, c1, c2, c3):
    x, y, z = state
    dx_dt = c1 * (y - x)
    dy_dt = x * (c2 - z) - y
    dz_dt = x * y - c3 * z
    return np.array([dx_dt, dy_dt, dz_dt])

def rk4_step(f, t, y, dt, c1, c2, c3):
    k1 = f(t, y, c1, c2, c3)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1, c1, c2, c3)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2, c1, c2, c3)
    k4 = f(t + dt, y + dt*k3, c1, c2, c3)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# Parameters
c1, c2, c3 = 10, 28, 8/3
t_start, t_end = 0, 400
dt = 0.001

delta_t = 0.1

initial_state = np.array([1.0, 1.0, 1.0])

# Time array
t = np.arange(t_start, t_end, dt)

# Initialize state array
states = np.zeros((len(t), 3))
states[0] = initial_state


# Solve using RK4
for i in range(1, len(t)):
    states[i] = rk4_step(lorenz_system, t[i-1], states[i-1], dt, c1, c2, c3)

# Extract X, Y, Z coordinates
X, Y, Z = states.T

x_sampled = X[::100]
y_sampled = Y[::100]
z_sampled = Z[::100]


# exporting sampled data to JSON

data = {'t': t[::100].tolist(), 'x': x_sampled.tolist(), 'y': y_sampled.tolist(), 'z': z_sampled.tolist()}
with open('lorentz_system_data.json', 'w') as jsonfile:
    json.dump(data, jsonfile)


# Plot X and Z coordinates over time
plt.figure(figsize=(12, 6))
plt.plot(t, X, label='X')
plt.plot(t, Z, label='Z')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Lorenz System: X and Z coordinates over time')
plt.legend()
plt.grid(True)
plt.show()

# 3D plot of the Lorenz attractor
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_sampled, y_sampled, z_sampled)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor')
plt.show()