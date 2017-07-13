from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.initializers import *
from keras.optimizers import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D




num_of_neurons = range(10,100,10)
d = np.zeros(len(num_of_neurons))
index = 0

# load data
x_data = np.loadtxt('x_data_set')
y_data = np.loadtxt('y_data_set')

split_ratio = 0.7
number_of_samples = np.shape(x_data)[0]

# train data
x_data_train = x_data[0: int(number_of_samples*split_ratio), ]
y_data_train = y_data[0: int(number_of_samples*split_ratio), ]

# test data
x_data_test = x_data[int(number_of_samples*split_ratio):, ]
y_data_test = y_data[int(number_of_samples*split_ratio):, ]

number_of_test_data = np.shape(x_data_test)[0]

# settings
seed_range = range(1, 310000, 30000)
number_of_seeds = len(seed_range)
lr_range = np.linspace(0.001, 0.1, 10)
number_of_learning_rate = len(lr_range)
d = np.zeros(len(lr_range))
num_of_seed = len(seed_range)
result = np.ndarray(shape=(num_of_seed, 3), dtype=float)

train_loss= np.zeros((number_of_seeds,number_of_learning_rate))


for j, _lr in enumerate(lr_range):
    adam = Adam(lr=_lr)
    for i, seed in enumerate(seed_range):
        np.random.seed(seed)
        model = Sequential()
        model.add(Dense(units=200, input_shape=(5,), kernel_initializer='random_normal', bias_initializer='random_normal',
                        activation='relu'))
        model.add(Dense(units=200, kernel_initializer='random_normal', bias_initializer='random_normal',
                        activation='sigmoid'))
        model.add(Dense(units=9, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
        model.compile(loss='mean_squared_error', optimizer=adam)
        history = model.fit(x_data_train, y_data_train, batch_size=512, validation_split=0.1,epochs=200)
        train_loss[i][j] = history.history['loss'][-1]


X = np.repeat(seed_range, len(lr_range))
Y = np.tile(lr_range, len(seed_range))
Z = np.reshape(train_loss,len(seed_range)*len(lr_range))

print(train_loss)
print(seed_range)
print(lr_range)
# plot the result,
# lr_range and seed_range must have the same length

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(X, Y, Z)

ax.set_xlabel('seed')
ax.set_ylabel('learning rate')
ax.set_zlabel('loss')
plt.show()

