import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from functools import reduce

d2 = json.load(open("result_net_structure.txt"))

learning_rate = d2['learning_rate']
seed = d2['seed']
drop_out_rate = d2['drop_out_rate']
number_of_neuron = d2['number_of_neuron']
number_of_hidden_layer = d2['number_of_hidden_layer']
train_loss = d2['train_loss']
val_loss = d2['val_loss']
accuracy = d2['accuracy']
metric2 = d2['metric2']


print('learning rate range is %s' % np.unique(learning_rate))
print('seed range is %s' % np.unique(seed))
print('drop_out_rate range is %s' % np.unique(drop_out_rate))
print('neuron range is %s' % np.unique(number_of_neuron))
print('hidden layer range is %s' % np.unique(number_of_hidden_layer))

# from 5d to 3d, with neuron and layer fixed
a = np.where(np.asarray(number_of_hidden_layer)==3)
b = np.where(np.asarray(number_of_neuron)==200)
e = np.intersect1d(a,b)

seed = np.asarray(seed)[e]
learning_rate = np.asarray(learning_rate)[e]
drop_out_rate = np.asarray(drop_out_rate)[e]
metric2 = np.asarray(metric2)[e]


length_lr = len(np.unique(learning_rate))
length_dr = len(np.unique(drop_out_rate))
# with respect to seed
m2_variance_matrix = np.zeros((length_lr,length_dr))
m2_average_matrix = np.zeros((length_lr,length_dr))

for i,lr in enumerate(np.unique(learning_rate)):
    for j,dr in enumerate(np.unique(drop_out_rate)):
        ind1 = np.where(learning_rate == lr)
        ind2 = np.where(drop_out_rate == dr)
        ind = np.intersect1d(ind1,ind2)
        m2 = metric2[ind]
        m2_variance = np.var(m2)
        m2_average = np.average(m2)
        m2_variance_matrix[i][j] = m2_variance
        m2_average_matrix[i][j] = m2_average


fig, ax = plt.subplots()
cax = ax.imshow(m2_average_matrix,aspect='auto',interpolation='nearest', extent=[0.001, 0.01, 0, 0.9])
cbar = fig.colorbar(cax)
cbar.set_label('Cbar Label Here')
plt.title('200 neurons, 3 hidden layers, metric2 heatmap')
plt.ylabel('dropout rate')
plt.xlabel('learning rate')
plt.show()