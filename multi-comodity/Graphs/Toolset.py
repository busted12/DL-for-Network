import numpy as np


def remove_dup_dataset(x, y):
    '''
    sort out duplicate data
    '''
    dim_x = np.shape(x)[1]
    xy = np.concatenate((x,y),axis=1)
    xy_new = np.vstack({tuple(row) for row in xy})
    x_new = np.hsplit(xy_new, np.array([dim_x]))[0]
    y_new = np.hsplit(xy_new, np.array([dim_x]))[1]

    return x_new, y_new


def calculate_node_flow(edge_flow,
                        start_nodes=[0, 0, 1, 1, 1, 2, 2, 3, 4],
                        end_nodes= [1, 2, 2, 3, 4, 3, 4, 4, 2]):
    '''
    given the flow of each edge, calculate the net flow of the nodes
    '''

    if (len(start_nodes) == len(end_nodes) == len(edge_flow)) is True:
        num_of_nodes = len(np.unique(start_nodes))
        node_flow = np.zeros(num_of_nodes)
        for i in range(num_of_nodes):
            index = 0
            for i2 in start_nodes:
                if i2 == i:
                    node_flow[i] += edge_flow[index]
                index += 1
            index = 0
            for i2 in end_nodes:
                if i2 == i:
                    node_flow[i] -= edge_flow[index]
                index += 1
        return node_flow
    else:
        raise ValueError('start_ nodes, end_nodes, capacity, cost should be list, and have same length')

def round_counter2(x, y):
    '''
    count how many predictions are within the round-up and off region up to 2,

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

