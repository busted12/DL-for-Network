from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import *
from keras.optimizers import *
import numpy as np
from Toolset import *
import matplotlib.pyplot as plt


num_of_neurons = range(10,100,10)
d = np.zeros(len(num_of_neurons))
index = 0

# load data
x_data = np.loadtxt(r'/home/chen/MA_python/dataset/x_data_set_no_dup')
y_data = np.loadtxt(r'/home/chen/MA_python/dataset/y_data_set_no_dup')

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
d = np.zeros(3)


adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00)


np.random.seed(20001)
model = Sequential()
model.add(Dense(units=200, input_shape=(5,), kernel_initializer='random_normal', bias_initializer='random_normal', activation='relu'))
model.add(Dense(units=200, kernel_initializer='random_normal', bias_initializer='random_normal', activation='sigmoid'))
model.add(Dense(units=9, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
model.compile(loss='mean_squared_error',optimizer=adam)
history = model.fit(x_data_train, y_data_train, batch_size=128 , epochs=200)
loss = model.evaluate(x_data_test, y_data_test)




plt.plot(d)
axes = plt.gca()
axes.set_ylim([0,30])
plt.xlabel('run the NN 10 times')
plt.ylabel('loss')
plt.show()

predition = model.predict(x_data_test)
