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

print('maximum accuracy is ', max(accuracy))
print('minimum m2 is ', min(metric2))




# from 5d to 3d, with neuron and layer fixed
a = np.where(np.asarray(learning_rate)==0.001)
b = np.where(np.asarray(drop_out_rate)==0.0)
e = np.intersect1d(a,b)

seed = np.asarray(seed)[e]
number_of_hidden_layer = np.asarray(number_of_hidden_layer)[e]
number_of_neuron = np.asarray(number_of_neuron)[e]
metric2 = np.asarray(metric2)[e]
accuracy = np.asarray(accuracy)[e]

length_neuron = len(np.unique(number_of_hidden_layer))
length_layer = len(np.unique(number_of_neuron))
# with respect to seed
m2_variance_matrix = np.zeros((length_neuron,length_layer))
m2_average_matrix = np.zeros((length_neuron,length_layer))

accu_variance_matrix = np.zeros((length_neuron,length_layer))
accu_average_matrix = np.zeros((length_neuron,length_layer))

for i,layer in enumerate(np.unique(number_of_hidden_layer)):
    for j,neuron in enumerate(np.unique(number_of_neuron)):
        ind1 = np.where(number_of_hidden_layer == layer)
        ind2 = np.where(number_of_neuron == neuron)
        ind = np.intersect1d(ind1,ind2)
        m2 = metric2[ind]
        print(neuron)
        m2_variance = np.var(m2)
        m2_average = np.average(m2)
        m2_variance_matrix[i][j] = m2_variance
        m2_average_matrix[i][j] = m2_average
        accu = accuracy[ind]
        accu_variance = np.var(accu)
        accu_average = np.average(accu)
        accu_variance_matrix[i][j] = accu_variance
        accu_average_matrix[i][j] = accu_average


print(accu_average_matrix)
fig, ax = plt.subplots()
cax = ax.imshow(accu_average_matrix,aspect='auto', interpolation='nearest',extent=[10, 200, 3, 1])
cbar = fig.colorbar(cax)
cbar.set_label('Cbar Label Here')
plt.title('learning rate=0.001, dropout = 0.4, metric2 heatmap')
plt.ylabel('dropout rate')
plt.xlabel('learning rate')
plt.show()