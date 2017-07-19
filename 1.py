import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

d2 = json.load(open("result.txt"))

learning_rate = d2['learning_rate']
seed = d2['seed']
drop_out_rate = d2['drop_out_rate']
train_loss = d2['train_loss']
val_loss = d2['val_loss']

#a = np.vstack((learning_rate,seed))

#print(a)
print(drop_out_rate)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cm = plt.cm.get_cmap('brg')
pic = ax.scatter(learning_rate, seed, drop_out_rate, c=val_loss, cmap=cm, s=300)
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Seed')
ax.set_zlabel('Dropout Rate')
fig.colorbar(pic)
plt.show()

