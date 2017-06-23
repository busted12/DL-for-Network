from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import *
import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt('x_data_set')
y = np.loadtxt('y_data_set')

def unique_rows(x):
    B = np.concatenate((x, x))
    x_new = np.asarray(list(i for i in set(map(tuple, B))))
    return x_new

a = [[1, 1, 2],
     [1, 1, 2],
     [2, 1, 1]]
b= unique_rows(a)
print(b)