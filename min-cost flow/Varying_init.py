from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import *
from keras.optimizers import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab


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

initializers = [random_normal(), random_uniform(), truncated_normal(), he_normal(), he_uniform(), lecun_uniform()]
initializers_name = ['random_normal', 'random_uniform()', 'truncated_normal', 'he_normal()', 'he_uniform()', 'lecun_uniform()']

seeds = range(1, 90001, 15000)
learning_rates = np.linspace(0.001, 0.005, 5)


for j, seed in enumerate(seeds):

    for learning_rate in learning_rates:

        for i, initializer in enumerate(initializers):
            np.random.seed(seed)
            model = Sequential()
            model.add(Dense(units=200, input_shape=(5,), kernel_initializer=initializer,
                            bias_initializer=initializer, activation='relu'))
            model.add(Dense(units=200, kernel_initializer=initializer, bias_initializer=initializer,
                            activation='sigmoid'))
            model.add(Dense(units=9, activation='relu', kernel_initializer=initializer,
                            bias_initializer=initializer))
            model.compile(loss='mean_squared_error', optimizer=adam(lr=learning_rate))
            history = model.fit(x_data_train, y_data_train, batch_size=512, validation_split=0.1,epochs=200)
            train_loss = history.history['loss']
            # plot the loss
            init_label = 'initializer is  ' + initializers_name[i]
            plt.plot(train_loss, label=init_label)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        title = 'seed is '+ str(seed) + ' learning rate is ' + str(learning_rate)
        plt.title(title)
        file_name = 'seed = ' + str(seed) + 'learning rate = ' + str(learning_rate)
        plt.savefig('/home/chen/MA_python/loss-vs-init/' + str(file_name) + '.png')
        plt.close()


plt.show()