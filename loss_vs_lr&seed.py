from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import *
from keras.optimizers import *
import numpy as np
import matplotlib.pyplot as plt


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

seed_range = range(1, 100000, 15000)
lr_range = np.linspace(0.001, 0.005, 5)
d = np.zeros(len(lr_range))
num_of_seed = len(seed_range)
result = np.ndarray(shape=(num_of_seed, 3), dtype=float)


plt.figure()

for j, _lr in enumerate(lr_range):
    adam = Adam(lr=_lr)
    for i, seed in enumerate(seed_range):
        np.random.seed(seed)
        model = Sequential()
        model.add(Dense(units=50, input_shape=(5,), kernel_initializer='random_normal', bias_initializer='random_normal',
                        activation='relu'))
        model.add(Dense(units=50, kernel_initializer='random_normal', bias_initializer='random_normal',
                        activation='sigmoid'))
        model.add(Dense(units=9, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
        model.compile(loss='mean_squared_error', optimizer=adam)
        history = model.fit(x_data_train, y_data_train, batch_size=128, epochs=20)
        train_loss = history.history['loss']
        seed_label = 'seed = ' + str(seed)
        plt.plot(train_loss, label=seed_label)

    title = 'learning rate is  ' + str(_lr)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(r'learning rate is {}.png'.format(_lr))
    plt.close()

plt.show()