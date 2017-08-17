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

a = np.vstack((learning_rate,seed,drop_out_rate,train_loss))

print(a)
print(np.shape(a))

b = np.transpose(a)
print(b)
print(np.shape(b))

e = np.unique(b[:,0])
print(e)
f = np.unique(b[:,1])

for i, j in zip(e,f):
    x = b[: ,2][np.where(b[:,0]==i and b[:,1]==j)]

print(x)
print(np.shape(x))
