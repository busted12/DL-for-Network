from __future__ import division
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import *

import Toolset

num_of_neurons = range(10,100,10)
d = np.zeros(len(num_of_neurons))
index = 0

# load data
x_data = np.loadtxt('/home/chen/MA_python/multi-comodity/graph3_3/x_data_set3')
y_data = np.loadtxt('/home/chen/MA_python/multi-comodity/graph3_3/y_data_set3')

print(np.shape(x_data))

x_new, y_new = Toolset.remove_dup_dataset(x_data, y_data)

print(np.shape(x_new))

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
model.add(Dense(units=200, input_shape=(6,), kernel_initializer='random_normal', bias_initializer='random_normal', activation='relu'))
model.add(Dense(units=200, kernel_initializer='random_normal', bias_initializer='random_normal', activation='sigmoid'))
model.add(Dense(units=3, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
model.compile(loss='mean_squared_error',optimizer=adam)
history = model.fit(x_data_train, y_data_train, batch_size=128, validation_split= 0.3 , epochs=200)
loss = model.evaluate(x_data_test, y_data_test)

predict = model.predict(x_data_test)

round_predict = np.round(predict)

counter = Toolset.check_same_row(round_predict, y_data_test)
accuracy = counter / number_of_test_data
print(accuracy)

diff_matrix = round_predict - y_data_test

metric3 = np.zeros(number_of_test_data)
for i in range(number_of_test_data):
    metric3[i] = np.sum(np.absolute(diff_matrix)[i]) / (np.sum(y_data_test[i]))

M3 = np.sum(metric3) / number_of_test_data

print(M3)



