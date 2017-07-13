import numpy as np


def sort_dataset(x, y):
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




