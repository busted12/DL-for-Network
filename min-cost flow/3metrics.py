from __future__ import division
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.initializers import *
from keras.optimizers import *
import numpy as np
from Toolset import *
import matplotlib.pyplot as plt
import json

# Network structure
# start_nodes = [0, 0, 1, 1, 1, 2, 2, 3, 4]
# end_nodes = [1, 2, 2, 3, 4, 3, 4, 4, 2]
# capacities = [15, 8, 20, 4, 10, 15, 5, 20, 4]
# unit_costs = [4, 4, 2, 2, 6, 1, 3, 2, 3]


# load data to file
#with open('sim.par') as file:
#    parameters = json.load(file)

learning_rate = 0.001
seed = 1
drop_out_rate = 0
number_of_hidden_layer = 2
number_of_neuron = 250

# load data
x_data = np.loadtxt('x_data_set_no_dup')
y_data = np.loadtxt('y_data_set_no_dup')

split_ratio = 0.7
number_of_samples = np.shape(x_data)[0]

# train data
x_data_train = x_data[0: int(number_of_samples * split_ratio), ]
y_data_train = y_data[0: int(number_of_samples * split_ratio), ]

# test data
x_data_test = x_data[int(number_of_samples * split_ratio):, ]
y_data_test = y_data[int(number_of_samples * split_ratio):, ]

number_of_test_data = np.shape(x_data_test)[0]
print(number_of_test_data)



loss_log = np.zeros(10)
val_loss_log = np.zeros(10)

adam = Adam(lr=learning_rate)

# construct model
np.random.seed(seed)
model = Sequential()
# input layer and the first hidden layer
model.add(Dense(units=int(number_of_neuron / (1 - drop_out_rate)), input_shape=(5,), kernel_initializer='random_normal',
                activation='relu'))
model.add(Dropout(rate=drop_out_rate))

nr_hidden = 1

while nr_hidden < number_of_hidden_layer:
    # hidden layer
    model.add(Dense(units=int(number_of_neuron / (1 - drop_out_rate)), kernel_initializer='random_normal',
                    activation='relu'))
    model.add(Dropout(rate=drop_out_rate))
    nr_hidden += 1

# add output layer
model.add(Dense(units=9, activation='relu', kernel_initializer='random_normal'))
model.compile(loss='mean_squared_error', optimizer=adam)
history = model.fit(x_data_train, y_data_train, validation_split=0.3, batch_size=512, epochs=200, verbose=1)

# check loss
# loss is loss for each epoch
loss = history.history['loss']
# val_loss is the validation loss for each epoch
val_loss = history.history['val_loss']

# plot the results
# plt.figure()
# plt.plot(loss, label='train_ loss')
# plt.plot(val_loss, label='validation loss')
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# axes = plt.gca()
# plt.savefigure('results.png')


# check how many answers are real after rounding

predict = model.predict(x_data_test)

round_predict = np.round(predict)
floor_predict = np.floor(predict)
ceil_predict = np.ceil(predict)
counter = check_same_row(round_predict, y_data_test)

# metric1
accuracy = counter / number_of_test_data

# metric2
diff_matrix = round_predict - y_data_test
sum_of_edge_capacity = 101
M2 = np.sum(np.absolute(diff_matrix)) / ((number_of_test_data) * sum_of_edge_capacity)

# metric3
metric3 = np.zeros(number_of_test_data)
for i in range(number_of_test_data):
    metric3[i] = np.sum(np.absolute(diff_matrix)[i]) / (np.sum(y_data_test[i]))

M3 = np.sum(metric3) / number_of_test_data

# counter2 gives the number of data, within the rounding range
counter2 = round_counter(predict, y_data_test)
print('exact right answer is: ' + str(counter))
print('Answer with in rounding range is: ' + str(counter2))

dev1 = deviation_counter(round_predict, y_data_test)




capacity_diff = capacity_excess(round_predict)
d = capacity_diff.max()
print('the biggest capacity excess is ' + str(d))




nr_feasible = check_feasible(round_predict, x_data_test)
print(' number of feasible solution is ' + str(nr_feasible))

# print out final loss
final_train_loss = loss[-1]
final_val_loss = val_loss[-1]
print('train loss after 200 epochs is ' + str(final_train_loss))
print('validation loss after 200 epochs is ' + str(final_val_loss))


# save result variables to txt file as dictionary

result_dict = {}

result_dict['train_loss'] = final_train_loss
result_dict['val_loss'] = final_val_loss
result_dict['seed'] = seed
result_dict['drop_out_rate'] = drop_out_rate
result_dict['learning_rate'] = learning_rate
result_dict['number_of_hidden_layer'] = number_of_hidden_layer
result_dict['number_of_neuron'] = number_of_neuron
result_dict['accuracy'] = accuracy
result_dict['M2'] = M2
result_dict['M3'] = M3
print(result_dict)
