from LP_solver import *
import numpy as np


def create_datset(number_of_trials):
    '''
    feed the solver with number_of_trials randomly generated arrays
    :param number_of_trials:
    :return:
    '''
    counter = 0
    x_data = np.zeros(6)
    y_data = np.zeros(4)
    for i in range(number_of_trials):
        flat_demand_matrix = np.random.randint(low=0, high=30, size=6)

        d = main(flat_demand_matrix)
        if d[0] == 1:
            counter = counter + 1
            x_data = np.vstack((x_data, flat_demand_matrix))
            y_data = np.vstack((y_data, d[1]))
            print counter

    return x_data[1:counter+1, ], y_data[1:counter+1, ]


x, y = create_datset(40000)

np.savetxt('x_data_set3', x)
np.savetxt('y_data_set3', y)
print x
print y