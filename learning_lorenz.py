##### ridge regression implementation ########


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
import statistics













# Load the data
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
buffer_data_1_y = x[:round(len(t)/8)]
train_data_t = t[round(len(t)/8):round(3*(len(t)/8))]
train_data_y = x[round(len(t)/8):round(3*(len(t)/8))]
buffer_data_2_t = t[round(3*(len(t)/8)):round(5*(len(t)/8))]
buffer_data_2_y = x[round(3*(len(t)/8)):round(5*(len(t)/8))]
test_data_t = t[round(5*(len(t)/8)):round(7*(len(t)/8))]
test_data_y = x[round(5*(len(t)/8)):round(7*(len(t)/8))]
buffer_data_3_t = t[round(7*(len(t)/8)):]
buffer_data_3_y = x[round(7*(len(t)/8)):]


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
V0 = 750e-3 + 1e-2*h_input_flattened


print(f"number of V0 time steps: ",len(h_input_flattened))






with open('I_values_lorenz.json', 'r') as jsonfile:
        data = json.load(jsonfile)

t = np.array(data['t'])
I_values = np.array(data['I'])







print(f"number of time steps:", len(I_values))
#print(I_values)
y_min = min(I_values)
y_max = max(I_values)
y_padding = (y_max - y_min) * 0.4*np.mean(I_values)
y_ticks = np.linspace(y_min - y_padding, y_max + y_padding, num=5)
y_ticks = [round(y, 6) for y in y_ticks]
yticks_labels = [f'{y * 1e3:.6f}' for y in y_ticks]

# axs[1].plot(I_values)
# axs[1].scatter(step_indices, [I_values[i] for i in step_indices], color='red', label='Step Indices')
# axs[1].set_xlabel(r't$[ns]$', fontsize=15)
# axs[1].set_ylabel(r'I$[mA]$', fontsize=15)
# axs[1].set_xticks([0, 10e-9/time_step, 20e-9/time_step, 30e-9/time_step, 40e-9/time_step, 50e-9/time_step, 60e-9/time_step, 70e-9/time_step, 80e-9/time_step, 90e-9/time_step, 100e-9/time_step])
# axs[1].set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'])
# axs[1].tick_params(axis='x', labelsize=14)
# axs[1].set_yticks(y_ticks)
# axs[1].set_yticklabels(yticks_labels)
# axs[1].grid(True)
# axs[1].legend(loc='upper right')



# plt.tight_layout()
# plt.show()

print(f"number of step indices: ", len(step_indices))
step_indices_training = step_indices[round(len(step_indices)/8)+30:round(3*len(step_indices)/8)+30]
step_indices_testing = step_indices[round(5*len(step_indices)/8)+30:7*round(len(step_indices)/8)+30]
I_values_training_read = [I_values[i] for i in step_indices_training]
print(f"length of I_train reads", len(I_values_training_read))
print(f"length of train data",len(train_data_t))

S_matrix_training = np.array(I_values_training_read).reshape(len(train_data_t), no_virtual_nodes)
S_training_1d = S_matrix_training.ravel()

print(f"length of training steps: ", len(step_indices_training))
print(f"length of testing steps: ", len(step_indices_testing))
I_values_testing_read = [I_values[i] for i in step_indices_testing]
print(f"length of I_test reads", len(I_values_testing_read))
S_matrix_testing = np.array(I_values_testing_read).reshape(len(test_data_t), no_virtual_nodes)
S_testing_1d = S_matrix_testing.ravel()





# fig, axs = plt.subplots(2, 1, figsize=(14, 15))

# axs[0].plot(I_values_training_read)
# axs[0].set_xlabel(r't$[ns]$', fontsize=15)
# axs[0].set_ylabel(r'I$[mA]$', fontsize=15)

# axs[1].plot(S_training_1d)

# plt.tight_layout()
# plt.show()










alpha_values = np.logspace(-15, -9, 100)
target_vector_train = x[round(len(t)/8)+1:round(3*len(t)/8) +1]
target_vector_test = x[round(5*len(t)/8)+1:round(7*len(t)/8)+1] 

nrmse_train = []
nrmse_test = []

for alpha in alpha_values:
   
    
    # Predictions for training and testing data
    S_transpose_S = np.matmul(np.transpose(S_matrix_training), S_matrix_training)
    
    identity_matrix = np.identity(S_transpose_S.shape[0])
    STS_p_lambda_I = S_transpose_S + alpha*identity_matrix

    inverse = np.linalg.inv(STS_p_lambda_I)
    weights = np.matmul(np.matmul(inverse, np.transpose(S_matrix_training)), target_vector_train)
    predictions_train = np.matmul(S_matrix_training, weights)
    predictions_test = np.matmul(S_matrix_testing, weights)






    
    
    
    # Calculate NRMSE for training data
    sum_1 = sum((target_vector_train[i] - predictions_train[i])**2 for i in range(len(target_vector_train)))
    NRMSE_1 = np.sqrt(sum_1 / (len(target_vector_train) * np.var(target_vector_train)))
    
    nrmse_train.append(NRMSE_1)

    sum_2 = sum((target_vector_test[i] - predictions_test[i])**2 for i in range(len(target_vector_test)))
    NRMSE_2 = np.sqrt(sum_2 / (len(target_vector_test) * np.var(target_vector_test)))

    nrmse_test.append(NRMSE_2)


    
    




plt.figure(figsize=(10, 6))
plt.plot(alpha_values, nrmse_train, label='Training NRMSE')
plt.plot(alpha_values, nrmse_test, label='Testing NRMSE')
plt.xlabel('Alpha')
plt.ylabel('NRMSE')
plt.title('NRMSE vs Alpha for Ridge Regression')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.legend()
plt.grid(True)
plt.savefig('ridge_regression_ex_down_lorenz.pdf')
plt.show()
