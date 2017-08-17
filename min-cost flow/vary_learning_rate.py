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


learning_rate = np.linspace(0.0001, 0.001, 10)
d = np.zeros(len(learning_rate))

for i, lr in enumerate(learning_rate):
    np.random.seed(1)
    adam = Adam(lr=lr)
    model = Sequential()
    model.add(Dense(units=200, input_shape=(5,), kernel_initializer='random_normal', bias_initializer='random_normal', activation='relu'))
    model.add(Dense(units=200, kernel_initializer='random_normal', bias_initializer='random_normal', activation='sigmoid'))
    model.add(Dense(units=9, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
    model.compile(loss='mean_squared_error',optimizer=adam)
    history = model.fit(x_data_train, y_data_train, validation_split=0.1, batch_size=128, epochs=2000)
    loss = model.evaluate(x_data_test, y_data_test)
    d = history.history['loss']
    plt.plot(d, label='learning rate is ' + str(lr))

axes = plt.gca()
axes.set_ylim([0,1])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
