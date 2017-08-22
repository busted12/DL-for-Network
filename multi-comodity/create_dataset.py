from LP_solver import *
import numpy as np
from Toolset import *


def create_datset(number_of_trials,graph):
    '''
    feed the solver with number_of_trials randomly generated arrays
    :param number_of_trials:
    :return:
    '''


    counter = 0
    x_data = np.zeros(12)
    y_data = np.zeros(12)
    for i in range(number_of_trials):
        demand_matrix = np.random.randint(10, 50, (4, 4))
        np.fill_diagonal(demand_matrix, 0)

        flat_demand_matrix = demand_matrix.ravel()
        demand_vector = flat_demand_matrix[np.nonzero(flat_demand_matrix)]

        d = main(demand_matrix, graph)
        if d[0] == 1:
            counter = counter + 1
            x_data = np.vstack((x_data, demand_vector))
            y_data = np.vstack((y_data, d[1]))
            print counter

    return x_data[1:counter+1, ], y_data[1:counter+1, ]


x, y = create_datset(40000,u'/home/chen/MA_python/multi-comodity/Graphs/4-6')
x_new, y_new = remove_dup_dataset(x,y)


np.savetxt(u'/home/chen/MA_python/multi-comodity/Dataset/4-6-x', x_new)
np.savetxt(u'/home/chen/MA_python/multi-comodity/Dataset/4-6-y', y_new)
