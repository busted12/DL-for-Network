from __future__ import division
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import *
import matplotlib.pyplot as plt
from datetime import datetime
from keras.layers.normalization import BatchNormalization

from Vlink_Reconf.Vlink import Toolset


class NeuralNetwork():
    def __init__(self, x_file, y_file):
        self.x_file = x_file
        self.y_file = y_file
        self.history = None
    # load data

    def run(self):
        x_data = np.loadtxt(self.x_file)
        y_data = np.loadtxt(self.y_file)

        input_dim = np.shape(x_data)[1]
        output_dim = np.shape(y_data)[1]

        split_ratio = 0.7
        number_of_samples = np.shape(x_data)[0]

        # train data
        x_data_train = x_data[0: int(number_of_samples*split_ratio), ]
        y_data_train = y_data[0: int(number_of_samples*split_ratio), ]

        # test data
        x_data_test = x_data[int(number_of_samples*split_ratio):, ]
        y_data_test = y_data[int(number_of_samples*split_ratio):, ]

        number_of_test_data = np.shape(x_data_test)[0]

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00)

        np.random.seed(52001)
        model = Sequential()
        model.add(Dense(units=200, input_shape=(input_dim,), kernel_initializer='glorot_normal', bias_initializer='zero', activation='relu'))
        #model.add(BatchNormalization(axis=-1, epsilon=0.02, momentum=0.97))
        model.add(Dense(units=200, kernel_initializer='glorot_normal', bias_initializer='zero', activation='relu'))
        #model.add(BatchNormalization(axis=-1, epsilon=0.02, momentum=0.97))
        model.add(Dense(units=output_dim, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zero'))
        model.compile(loss='mean_squared_error',optimizer=adam)
        history = model.fit(x_data_train, y_data_train, batch_size=128, validation_split= 0.3 , epochs=2000)
        self.history = history
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
        return  accuracy

    def plot(self):
        plt.plot(self.history.history['val_loss'])
        plt.plot(self.history.history['loss'])
        plt.show()

# x1 = u'/home/chen/MA_python/multi-comodity/Dataset/4-6-2-x'
# y1 = u'/home/chen/MA_python/multi-comodity/Dataset/4-6-2-y'
#
# nn = NeuralNetwork(x1,y1)
# a1 = nn.run()
#
# x2 = u'/home/chen/MA_python/multi-comodity/Dataset/4-5-1-x'
# y2 = u'/home/chen/MA_python/multi-comodity/Dataset/4-5-1-y'
#
# nn = NeuralNetwork(x2,y2)
# a2 = nn.run()
#
#
# x3 = u'/home/chen/MA_python/multi-comodity/Dataset/4-5-2-x'
# y3 = u'/home/chen/MA_python/multi-comodity/Dataset/4-5-2-y'
#
# nn = NeuralNetwork(x3,y3)
# a3 = nn.run()

x3 = u'/home/chen/MA_python/multi-comodity/Dataset/7-8-x'
y3 = u'/home/chen/MA_python/multi-comodity/Dataset/7-8-y'


start_time = datetime.now()

nn = NeuralNetwork(x3,y3)
a3 = nn.run()

end_time = datetime.now()
duration = end_time - start_time


print('7-6Graph', a3)
print duration
