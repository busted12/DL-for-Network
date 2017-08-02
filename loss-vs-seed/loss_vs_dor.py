from __future__ import division
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.initializers import *
from keras.optimizers import *
import numpy as np
from Toolset import *
import matplotlib.pyplot as plt


# Network structure
# start_nodes = [0, 0, 1, 1, 1, 2, 2, 3, 4]
# end_nodes = [1, 2, 2, 3, 4, 3, 4, 4, 2]
# capacities = [15, 8, 20, 4, 10, 15, 5, 20, 4]
# unit_costs = [4, 4, 2, 2, 6, 1, 3, 2, 3]

# load data
x_data = np.loadtxt(r'/home/chen/MA_python/networkX/x_data_set2')
y_data = np.loadtxt(r'/home/chen/MA_python/networkX/y_data_set2')

split_ratio = 0.7
number_of_samples = np.shape(x_data)[0]

# train data
x_data_train = x_data[0: int(number_of_samples*split_ratio), ]
y_data_train = y_data[0: int(number_of_samples*split_ratio), ]

# test data
x_data_test = x_data[int(number_of_samples*split_ratio):, ]
y_data_test = y_data[int(number_of_samples*split_ratio):, ]

number_of_test_data = np.shape(x_data_test)[0]
print(number_of_test_data)

loss_log = np.zeros(10)
val_loss_log = np.zeros(10)

adam = Adam(lr=0.005)

drop_out_rates = np.arange(0, 1, 0.1)

drop_out_rate = 0
np.random.seed(1)
model = Sequential()
model.add(Dense(units=int(100/(1-drop_out_rate)), input_shape=(10,), kernel_initializer='random_normal', activation='relu'))
model.add(Dropout(rate=drop_out_rate))
model.add(Dense(units=int(100/(1-drop_out_rate)), kernel_initializer='random_normal', activation='sigmoid'))
model.add(Dropout(rate=drop_out_rate))
model.add(Dense(units=int(100/(1-drop_out_rate)), kernel_initializer='random_normal', activation='relu'))
model.add(Dropout(rate=drop_out_rate))
model.add(Dense(units=int(100/(1-drop_out_rate)), kernel_initializer='random_normal', activation='relu'))
model.add(Dropout(rate=drop_out_rate))
model.add(Dense(units=36, activation='relu', kernel_initializer='random_normal'))
model.compile(loss='mean_squared_error',optimizer=adam)
history = model.fit(x_data_train, y_data_train, validation_split=0.28, batch_size=512, epochs=2000, verbose=1)

# check loss
# loss is loss for each epoch
loss = history.history['loss'][-1]
# val_loss is the validation loss for each epoch



# plot train loss and validation loss
plt.plot(drop_out_rates, loss_log, label='train_ loss')
plt.plot(drop_out_rates, val_loss_log, label='validation loss')
plt.legend()
plt.xlabel('drop out rate')
plt.ylabel('loss')
axes = plt.gca()
plt.savefig('Dropout_rate_vs_loss_with_pn=constant_val_split=0.28.png')
plt.close()


# check how many answers are real after rounding

predict=model.predict(x_data_test)

round_predict = np.round(predict)
floor_predict = np.floor(predict)
ceil_predict = np.ceil(predict)
counter = check_same_row(round_predict, y_data_test)
# counter2 gives the number of data, within the rounding range
counter2 = round_counter(predict, y_data_test)
print('exact right answer is: ' + str(counter))
print('Answer with in rounding range is: ' + str(counter2))

dev1 = deviation_counter(round_predict, y_data_test)


# calculate the node flow for each case(row-wise manipulation)

node_flow_all = calculate_node_for_all(round_predict, 10)

# check how many case where node flow are the same
counter3 = check_same_row(node_flow_all, x_data_test)
print(str(counter3) + ' cases, the conversation rule is not hurt')



capacity_diff = capacity_excess(round_predict)
d = capacity_diff.max()
print('the biggest capacity excess is ' + str(d))



nr_feasible = check_feasible(round_predict, x_data_test)
print('number of feasible answer is:', nr_feasible)


# plot
plt.bar(range(len(dev1)),dev1)
plt.xlabel('deviation')
plt.ylabel('number of samples')
plt.show()