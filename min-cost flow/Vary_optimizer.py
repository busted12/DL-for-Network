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

optimizers = [adam(lr=0.005), adagrad(lr=0.005), adamax(lr=0.005), adadelta(), rmsprop(lr=0.005)]
optimizers_name = ['adam', 'adagrad', 'adamax', 'adadelta', 'rmsprop']

seeds = range(1,90001,15000)


for j, seed in enumerate(seeds):
    for i, optimizer in enumerate(optimizers):
        np.random.seed(seed)
        model = Sequential()
        model.add(Dense(units=200, input_shape=(5,), kernel_initializer='random_normal', bias_initializer='random_normal',
                        activation='relu'))
        model.add(Dense(units=200, kernel_initializer='random_normal', bias_initializer='random_normal',
                        activation='sigmoid'))
        model.add(Dense(units=9, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        history = model.fit(x_data_train, y_data_train, batch_size=128, validation_split=0.1,epochs=200)
        train_loss = history.history['loss']
        optimizers_label = 'optimizer is ' + optimizers_name[i]
        plt.plot(train_loss, label=optimizers_label)

    title = 'seed is  ' + str(seed)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(r'loss-vs-optimizer/seed=' + str(seed)+'.png')
    plt.close()

