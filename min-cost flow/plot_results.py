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
train_loss = d2['train_loss']
val_loss = d2['val_loss']
accuracy = d2['accuracy']
metric2 = d2['metric2']
number_of_neuron = d2['number_of_neuron']
number_of_hidden_layer = d2['number_of_hidden_layer']


result_matrix = np.vstack((learning_rate,seed,drop_out_rate, number_of_neuron, number_of_hidden_layer,
                           train_loss,val_loss, accuracy, metric2))

print(np.shape(result_matrix))


# 5 variables
unique_learning_rate = np.unique(result_matrix[0,: ])
unique_seed = np.unique(result_matrix[1,: ])
unique_dropout_rate = np.unique(result_matrix[2,: ])
unique_number_of_neuron = np.unique(result_matrix[3,: ])
unique_number_of_hidden_layer = np.unique(result_matrix[4,: ])
a = np.where(result_matrix[1,: ] == unique_learning_rate[0])
b = np.where(result_matrix[2,: ] == unique_dropout_rate[0])
c = np.where(result_matrix[3,: ] == unique_number_of_neuron[3])
d = np.where(result_matrix[4,: ] == unique_number_of_hidden_layer[0])
e = reduce(np.intersect1d,(a,b,c,d))

print(e)

X = result_matrix[0,: ][e]
Y = result_matrix[8,: ][e]

f = np.argmax(result_matrix[7,: ])
print(f)
print('max accuracy is', result_matrix[7,: ][f])
print('lr=' , result_matrix[0,: ][f])
print('seed=' , result_matrix[1,: ][f])
print('dropout rate=' , result_matrix[2,: ][f])
print('neurons =' , result_matrix[3,: ][f])
print('layer =' , result_matrix[4,: ][f])

index = X.argsort()
sortedY = Y[index]
sortedX = X[index]

plt.plot(sortedX, sortedY)
title = 'lr=%s, dropout=%s,%s neurons, %s hidden layer' % (unique_learning_rate[0],unique_dropout_rate[0],
                                                           unique_number_of_neuron[0],unique_number_of_hidden_layer[0])
plt.title(title)
plt.show()

#a = np.vstack((learning_rate,seed))

#print(a)
#print(drop_out_rate)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
'''
4D heat map plot
cm = plt.cm.get_cmap('brg')
pic = ax.scatter(learning_rate, seed, drop_out_rate, c=val_loss, cmap=cm, s=300)
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Seed')
ax.set_zlabel('Dropout Rate')
cb =fig.colorbar(pic)
cb.set_label('validation loss')
plt.show()
'''