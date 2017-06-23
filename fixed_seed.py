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

# set initializers
W1 = random_normal(seed=1)
W2 = random_uniform(seed=100000)

# number of iterations for a given seed
seed_iteration = 5

d = np.zeros(seed_iteration)
index = 0
seed_range = range(1,100000,10000)
num_of_seed = len(seed_range)
result = np.ndarray(shape=(num_of_seed,seed_iteration),dtype=float)

adam = Adam(lr=0.01)

fig =  plt.figure()

for j, seed in enumerate(seed_range):
    for i in range(seed_iteration):
        np.random.seed(seed)
        model = Sequential()
        model.add(Dense(units=200, input_shape=(5,), kernel_initializer='random_normal', bias_initializer='random_normal', activation='relu'))
        model.add(Dense(units=200, kernel_initializer='random_normal', bias_initializer='random_normal', activation='sigmoid'))
        model.add(Dense(units=9, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
        model.compile(loss='mean_squared_error',optimizer=adam)
        history = model.fit(x_data_train, y_data_train, batch_size=128, epochs= 20)
        loss = model.evaluate(x_data_test, y_data_test)
        d[i] = history.history['loss'][-1]
    result[j] = d

    plt.plot(result[j])

np.savetxt('varying seed, epoch = 50',result)


axes = plt.gca()
axes.set_ylim([0,50])
plt.xlabel('run the NN 5 times')
plt.ylabel('loss')
plt.show()

test = model.evaluate(x_data_test,y_data_test)
print('test loss is ', test)
