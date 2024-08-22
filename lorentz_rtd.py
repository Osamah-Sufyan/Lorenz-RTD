import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json






class DifferentialEquationSolver:
    def __init__(self, params):
        self.a = params['a']
        self.q_e = params['q_e']
        self.k_B = params['k_B']
        self.T = params['T']
        self.b = params['b']
        self.c = params['c']
        self.d = params['d']
        self.n1 = params['n1']
        self.n2 = params['n2']
        self.h = params['h']
        self.R = params['R']
        self.L = params['L']
        self.C = params['C']
        self.kappa = params['kappa']
        self.gamma_m = params['gamma_m']
        self.gamma_l = params['gamma_l']
        self.gamma_nr = params['gamma_nr']
        self.eta = params['eta']
        self.q_e = params['q_e']
        self.J = params['J']
        self.N0 = params['N0']
        self.tau_p = params['tau_p']
        self.V0 = params['V0']
    # def Vm(self,V0, t):
    #     if 0.05e-9 <= t <= 0.1e-9:  # replace t_start and t_end with your interval
    #         return V0 # constant amplitude pulse
    #     else:
    #         return 750e-3 # no pulse outside the interval
    def Vm(self, V0, t):
        dt = 6000e-9 / 6000000
        index = int(t / dt)
        if index < 0:
            index = 0
        elif index >= len(V0):
            index = len(V0) - 1
        return V0[index]
        

    def f(self, V):
        term1 = (1 + np.exp(self.q_e * (self.b - self.c + self.n1 * V) / (self.k_B * self.T))) / \
                (1 + np.exp(self.q_e * (self.b - self.c - self.n1 * V) / (self.k_B * self.T)))
        term2 = np.pi / 2 + np.arctan((self.c - self.n1 * V) / self.d)
        term3 = self.h * (np.exp(self.q_e * self.n2 * V / (self.k_B * self.T)) - 1)
        return self.a * np.log(term1) * term2 + term3

    def system_derivs(self, y, t):
        V, I= y
        term1 = (1 + np.exp(self.q_e * (self.b - self.c + self.n1 * V) / (self.k_B * self.T))) / \
                (1 + np.exp(self.q_e * (self.b - self.c - self.n1 * V) / (self.k_B * self.T)))
        term2 = np.pi / 2 + np.arctan((self.c - self.n1 * V) / self.d)
        term3 = self.h * (np.exp(self.q_e * self.n2 * V / (self.k_B * self.T)) - 1)
        fV = self.a * np.log(term1) * term2 + term3
        
        dVdt = (I - fV )/(self.C)# Simplified m(t)
        dIdt = (self.Vm(self.V0, t) - V - self.R * I )/(self.L) 
        # dSdt = (self.gamma_m * (N - self.N0) - 1 / self.tau_p) * S + self.gamma_m * N +np.sqrt(self.gamma_m*N*S) * np.random.normal(0, 1)
        # dNdt = (self.J + self.eta * I) / self.q_e - (self.gamma_l + self.gamma_m + self.gamma_nr) * N - self.gamma_m * (N - self.N0) * S
        return np.array([dVdt, dIdt])

    def rk4_step(self, y, t, dt):
        k1 = dt * self.system_derivs(y, t)
        k2 = dt * self.system_derivs(y + 0.5 * k1, t + 0.5 * dt)
        k3 = dt * self.system_derivs(y + 0.5 * k2, t + 0.5 * dt)
        k4 = dt * self.system_derivs(y + k3, t + dt)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve(self, y0, t0, t_final, dt):
        num_steps = int((t_final - t0) / dt)
        
        t = t0
        y = y0
        results = []
        I_values = []
        V_values = []
        for _ in range(num_steps):
            results.append(y)
            I_values.append(y[1]) 
            V_values.append(y[0]) 
            y = self.rk4_step(y, t, dt)
            t += dt
        return np.array(results), np.array(I_values), np.array(V_values)

    def plot_fV(self, V_min, V_max):
        V_values = np.linspace(V_min, V_max, 400)
        fV_values = [self.f(V) for V in V_values]
        plt.figure(figsize=(10, 5))
        plt.plot(V_values, fV_values, label='f(V)')
        plt.xlabel('Voltage V')
        plt.ylabel('f(V)')
        plt.title('Plot of f(V)')
        plt.legend()
        plt.grid(True)
        plt.show()


# import json data from file

with open('lorentz_system_data.json', 'r') as jsonfile:
    data = json.load(jsonfile)
    t = np.array(data['t'])
    x = np.array(data['x'])
    y = np.array(data['y'])
    z = np.array(data['z'])

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Lorenz Attractor')
# plt.savefig('lorentz_attractor.pdf')
# plt.show()

print (len(x))

buffer_data_1_t = t[:round(len(t)/8)]
buffer_data_1_x = x[:round(len(t)/8)]
train_data_t = t[round(len(t)/8):round(len(t)/2)]
train_data_x = x[round(len(t)/8):round(len(t)/2)]
buffer_data_2_t = t[round(len(t)/2):round(5*len(t)/8)]
buffer_data_2_x = x[round(len(t)/2):round(5*len(t)/8)]

test_data_t = t[round(5*len(t)/8):]
test_data_x = x[round(5*len(t)/8):]
print(f"total divisions: ",len(buffer_data_1_t)+len(train_data_t)+len(buffer_data_2_t)+len(test_data_t))
x_min = np.min(x)-0.1
x_max = np.max(x)+0.1
no_virtual_nodes = 30
random_heights = np.random.uniform(0, 1, no_virtual_nodes)
int_len = t[1]-t[0]
mask = np.linspace(t[0]-int_len, t[0]+ int_len, 1500)  



print(f"number of Lorenz time steps: ",len(t))


h = np.zeros(len(mask)) 
line_width = len(mask)/no_virtual_nodes
print(f"len(mask):",len(mask))
print(f"line width:",line_width)

for j, height in enumerate(random_heights):
    x_start = int(j * line_width)
    x_end = int(x_start + line_width)
    for i in range(x_start, x_end):
        h[i] = height

num_raw_data = len(t)
h_rep = np.tile(h, num_raw_data)
h_repeated = np.repeat(h[np.newaxis, :], num_raw_data, axis=0)
h_input = h_repeated
for i in range(num_raw_data):
    h_input[i] = h_repeated[i] * x[i] 

h_input_flattened = h_input.flatten() 
  
x_repeated = np.linspace(t[0] - int_len, t[0] + num_raw_data * int_len, len(h_input_flattened))
print(len(h_input_flattened))
step_indices = []



for i in range(0, len(h_input_flattened)):
    if h_input_flattened[i] != h_input_flattened[i - 1]:
        step_indices.append(i)


print(f"number step indices: ",len(step_indices))
V0 = 750e-3 + 1e-3*h_input_flattened



print(f"number of V0 time steps: ",len(h_input_flattened))
params = {
    'a': -5.5e-5, 'q_e': 1.6e-19, 'k_B': 1.38e-23, 'T': 300,
    'b': 0.033, 'c': 0.113, 'd': -2.8e-6, 'n1': 0.185, 'n2': 0.045, 'h': 18e-5,
    'R': 10, 'kappa': 1.1e-7, 'gamma_m': 1e7, 'gamma_l': 1e9, 'gamma_nr': 2e9,
    'eta': 1, 'J': 210e-6, 'N0': 5e5, 'tau_p': 5e-13, 'L': 126e-9, 'C': 2e-15, 'V0': V0
}

solver = DifferentialEquationSolver(params)
initial_conditions = [750e-3, 75e-6]

time_step = 6000e-9  / len(h_input_flattened)
print(f"time step: ", time_step)


results, I_values, V_values = solver.solve(initial_conditions, 0e-10, 6000e-9, time_step )


data = {'t': t.tolist(), 'I': I_values.tolist()}
with open('I_values_lorenz.json', 'w') as jsonfile:
    json.dump(data, jsonfile)

data = {'t': t.tolist(), 'h': h_input_flattened.tolist()}
with open('h_values_lorenz.json', 'w') as jsonfile:
    json.dump(data, jsonfile)

print(f"number of time steps:", len(I_values))
print(I_values)
y_min = min(I_values)
y_max = max(I_values)

fig, axs = plt.subplots(2, 1, figsize=(12, 12))

axs[0].plot(x_repeated, h_input_flattened, label='masked input')
axs[0].scatter(t, y, color='red', label='mackey glass')
axs[0].set_xlabel(r'$t_{mg}$')
axs[0].legend(loc='upper right')

axs[1].plot(I_values)
axs[1].scatter(step_indices, [I_values[i] for i in step_indices], color='red', label='Step Indices')
axs[1].set_xlabel(r't$[ns]$', fontsize=15)
axs[1].set_ylabel(r'I$[mA]$', fontsize=15)
# axs[1].set_xticks([0, 10e-9/time_step, 20e-9/time_step, 30e-9/time_step, 40e-9/time_step, 50e-9/time_step, 60e-9/time_step, 70e-9/time_step, 80e-9/time_step, 90e-9/time_step, 100e-9/time_step])
# axs[1].set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'])
# axs[1].tick_params(axis='x', labelsize=14)
# axs[1].set_yticks(y_ticks)
# axs[1].set_yticklabels(yticks_labels)
axs[1].grid(True)
axs[1].legend(loc='upper right')



plt.tight_layout()
plt.show()



