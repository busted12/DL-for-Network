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
print(number_of_test_data)


def check_same_row(x, y):
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


def round_counter(x, y):
    '''
    count how many predictions are within the round-up and off region,

    :param x: n * m array, n is the number of examples.
    :param y: n * m array, n is the number of examples.
    :return: counter
    '''

    if np.shape(x) == np.shape(y):
        num_of_rows = np.shape(x)[0]
        floored = np.floor(x)
        diff = np.subtract(y, floored)
        counter = 0
        for row in diff:
            if np.max(row) >= 0 and np.max(row) <= 1:
                counter += 1
        return counter
    else:
        raise ValueError('x and y must have same shape')


def round_counter2(x, y):
    '''
    count how many predictions are within the round-up and off region,

    :param x: n * m array, n is the number of examples.
    :param y: n * m array, n is the number of examples.
    :return: counter
    '''

    if np.shape(x) == np.shape(y):
        num_of_rows = np.shape(x)[0]
        floored = np.floor(x)
        diff = np.subtract(y, floored)
        counter = 0
        for row in diff:
            if np.max(row) >= 1 and np.max(row) <= 3:
                counter += 1
        return counter
    else:
        raise ValueError('x and y must have same shape')


loss_log = np.zeros(9)
val_loss_log = np.zeros(9)

adam = Adam(lr=0.003)

drop_out_rates = np.arange(0.1, 1, 0.1)

for i, drop_out_rate in enumerate(drop_out_rates):
    np.random.seed(1)
    model = Sequential()
    model.add(Dense(units=int(200/drop_out_rate), input_shape=(5,), kernel_initializer='random_normal', activation='relu'))
    model.add(Dropout(rate=drop_out_rate))
    model.add(Dense(units=int(200/drop_out_rate), kernel_initializer='random_normal', activation='sigmoid'))
    model.add(Dropout(rate=drop_out_rate))
    model.add(Dense(units=9, activation='relu', kernel_initializer='random_normal'))
    model.compile(loss='mean_squared_error',optimizer=adam)
    history = model.fit(x_data_train, y_data_train, validation_split=0.1, batch_size=512, epochs=20, verbose=1)

    # check loss
    # loss is loss for each epoch
    loss = history.history['loss'][-1]
    # val_loss is the validation loss for each epoch
    val_loss = history.history['val_loss'][-1]
    loss_log[i] = loss
    val_loss_log[i] = val_loss


# plot train loss and validation loss
plt.plot(drop_out_rates, loss_log, label='train_ loss')
plt.plot(drop_out_rates, val_loss_log, label='validation loss')
plt.legend()
plt.xlabel('drop out rate')
plt.ylabel('loss')
axes = plt.gca()
plt.savefig('Dropout rate is 0.5.png')
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
def calculate_node_for_all(x, nr_of_nodes):
    '''
    give a set of edge flow, calculate its correspdonding node flow
    :param x: x is a n by m matrix, where each row is the edge flow of a specific case
    :return:  set of node flow
    '''

    node_flow_all = np.zeros(nr_of_nodes)
    for row in x:
        node_flow = calculate_node_flow(row)
        node_flow_all = np.vstack((node_flow_all, node_flow))
    node_flow_all = node_flow_all[1:, ]
    return node_flow_all

node_flow_all = calculate_node_for_all(round_predict, 5)

# check how many case where node flow are the same
counter3 = check_same_row(node_flow_all, x_data_test)
print(str(counter3) + ' cases, the conversation rule is not hurt')


# calculate the number
def capacity_excess(x, edge_capacities=[15, 8, 20, 4, 10, 15, 5, 20, 4]):
    '''
    give a set of edge flow, calculate its correspdonding capacity excess.
    :param x: x is a n by m matrix, where each row is the edge flow of a specific case
    :param edge_capacities:
    :return:
    '''
    diff_matrix = np.zeros(9)
    for row in x:
        diff = np.subtract(row, edge_capacities)
        diff_matrix = np.vstack((diff_matrix, diff))
    diff_matrix = diff_matrix[1:, ]
    return diff_matrix

capacity_diff = capacity_excess(round_predict)
d = capacity_diff.max()
print('the biggest capacity excess is ' + str(d))


def check_feasible(edge_flows, real_node_flows, edge_capacities=[15, 8, 20, 4, 10, 15, 5, 20, 4]):
    '''
    edge_flow and real_node_flow should have same number of rows, which means same data points.
    give a set of edge flow, calculate how many cases are feasible solutions. i.e
    1. capacity constraint is satisfied.
    2. node flow is satisfied
    note: node flow is calculated given edge flow and network structure
    :param x: x is a n by m matrix, where each row is the edge flow of a specific case
    :param edge_capacities:
    :return:
    '''

    counter = 0
    for edge_flow, real_node_flow in zip(edge_flows, real_node_flows):
        node_flow = calculate_node_flow(edge_flow)
        capacity_diff = edge_flow-edge_capacities
        if np.array_equal(node_flow,real_node_flow) and capacity_diff.max()<=0:
            counter += 1
    return counter

nr_feasible = check_feasible(round_predict, x_data_test)
print('number of feasible answer is:', nr_feasible)


# plot
plt.bar(range(len(dev1)),dev1)
plt.xlabel('deviation')
plt.ylabel('number of samples')
plt.show()