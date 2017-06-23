from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import *
import numpy as np
from Toolset import *
import matplotlib.pyplot as plt


num_of_neurons = range(10,100,10)
d = np.zeros(len(num_of_neurons))
index = 0

# load data
x_data = np.loadtxt(r'/home/chen/MA_python/dataset/x_data_set')
y_data = np.loadtxt(r'/home/chen/MA_python/dataset/y_data_set')

split_ratio = 0.7
number_of_samples = np.shape(x_data)[0]

# train data
x_data_train = x_data[0: int(number_of_samples*split_ratio), ]
y_data_train = y_data[0: int(number_of_samples*split_ratio), ]

# test data
x_data_test = x_data[int(number_of_samples*split_ratio):, ]
y_data_test = y_data[int(number_of_samples*split_ratio):, ]

number_of_test_data = np.shape(x_data_test)[0]

# set initializers
W1 = random_normal(seed=1)
W2 = random_uniform(seed=100000)

d = np.zeros(5)
index = 0
seed_range = range(1,50000,10000)
for i in seed_range:
    W1 = random_normal(seed=i)
    model = Sequential()
    model.add(Dense(units=200, input_shape=(5,), kernel_initializer=W1, bias_initializer=W1, activation='relu'))
    model.add(Dense(units=200, kernel_initializer=W2, bias_initializer=W1, activation='sigmoid'))
    model.add(Dense(units=9, activation='relu', kernel_initializer=W1, bias_initializer=W1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(x_data_train, y_data_train, batch_size=128, epochs= 50)
    loss = model.evaluate(x_data_test, y_data_test)
    d[index] = loss
    index += 1

plt.plot(d)
plt.xlabel(str(i) + 'th training')
plt.ylabel('loss')
plt.show()

predition = model.predict(x_data_test)

rounded_prediction = np.round(predition)


