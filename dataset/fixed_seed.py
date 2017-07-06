from __future__ import division
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import *
from keras.optimizers import *
import numpy as np
from Toolset import *
import matplotlib.pyplot as plt


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


# number of iterations for a given seed
seed_iteration = 5

d = np.zeros(seed_iteration)
index = 0
seeds = range(1, 90001, 15000)


learning_rate = np.linspace(0.01, 0.1, 10)


def check_result(x, y):
    '''
    compare two array row wise and return how many rows are exactly the same
    :param x: adarray
    :param y: adarray
    :return: counter
    '''
    if np.shape(x) == np.shape(y):
        counter = 0
        for j in range(np.shape(x)[0]):
            if np.array_equal(x[j], y[j]) is True:
                counter += 1
        return counter
    else:
        raise ValueError('x and y must be same shape')


def deviation_counter(x, y):
    '''
    compare two ndarray row wise and check how big the deviation is, return is an
    array with as the number of sample of deviation.
    for example. x = [[1,1,1],[1,1,2],[1,1,3]], y = [[1, 1, 1], [1,2,1],[1,1,2]]
    return should a dictionary: a = {1, 1, 1}, which means 0 for once, 1 for once,
    2 for once. index is the deviation.
    :param x:
    :param y:
    :return:
    '''
    if np.shape(x) == np.shape(y):
        num_of_rows = np.shape(x)[0]
        diff = np.subtract(x, y)
        abs_diff = np.absolute(diff)
        abs_diff_sum = np.sum(abs_diff, axis=1, dtype=np.int64)   # row-wise add up the deviation
        deviation_count = np.bincount(abs_diff_sum)
        return deviation_count
    else:
        raise ValueError('x and y must have same shape')


loss_log = np.zeros(10)
val_loss_log = np.zeros(10)



np.random.seed(1)
model = Sequential()
model.add(Dense(units=100, input_shape=(5,), kernel_initializer='random_normal', activation='relu'))
model.add(Dense(units=100, kernel_initializer='random_normal', activation='sigmoid'))
model.add(Dense(units=9, activation='relu', kernel_initializer='random_normal'))
model.compile(loss='mean_squared_error',optimizer='adam')
history = model.fit(x_data_train, y_data_train, validation_split=0.1, batch_size=512, epochs=2000)

# check loss
# loss is loss for each epoch
loss = history.history['loss']
# val_loss is the validation loss for each epoch
val_loss = history.history['val_loss']
plt.plot(loss, label='train_ loss')
plt.plot(val_loss, label='validation loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
axes = plt.gca()
plt.close()

# check how many answers are real after rounding

predict=model.predict(x_data_test)
round_predict = np.round(predict)
counter = check_result(round_predict, y_data_test)
dev1 = deviation_counter(round_predict, y_data_test)

plt.bar(range(len(dev1)),dev1)
plt.xlabel('deviation')
plt.ylabel('number of samples')
plt.show()









