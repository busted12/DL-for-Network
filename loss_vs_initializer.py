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

seed = 50
weight_init = ['random_normal', 'zeros', 'random_uniform', 'truncated_normal']
lr_range = np.linspace(0.0005, 0.005, 5)
d = np.zeros(len(weight_init))
plt.figure()

for j, _lr in enumerate(lr_range):
    adam = Adam(lr=_lr)
    for i, initializer in enumerate(weight_init):
        np.random.seed(seed)
        model = Sequential()
        model.add(Dense(units=50, input_shape=(5,), kernel_initializer=weight_init[i], bias_initializer=weight_init[i],
                        activation='relu'))
        model.add(Dense(units=50, kernel_initializer=weight_init[i], bias_initializer=weight_init[i],
                        activation='sigmoid'))
        model.add(Dense(units=9, activation='relu', kernel_initializer=weight_init[i], bias_initializer=weight_init[i]))
        model.compile(loss='mean_squared_error', optimizer=adam)
        history = model.fit(x_data_train, y_data_train, batch_size=128, epochs= 20)
        train_loss = history.history['loss']
        lr_label = 'weight initializer is ' + str(weight_init[i])
        plt.plot(train_loss, label=lr_label)

    title = 'learning rate = ' + str(_lr)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('varying weight initializer,learning rate is _{}.png'.format(_lr))
    plt.close()
plt.show()