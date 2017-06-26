import numpy as np

def calculate_node_flow(edge_flow, start_nodes=[0, 0, 1, 1, 1, 2, 2, 3, 4],
                         end_nodes=[1, 2, 2, 3, 4, 3, 4, 4, 2]):
    if (len(start_nodes) == len(end_nodes) == len(edge_flow)) is False:
        raise ValueError('start_ nodes, end_nodes, capacity, cost should be list, and have same length')
    else:
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

node_flow = calculate_node_flow(edge_flow=[0,5,   3,   4,   0,   2,   0,  14,   0])
print(node_flow)


def sort_dataset(x):
    x_new = np.vstack({tuple(row) for row in x})
    return x_new

x = np.loadtxt('x_data_set')
y = np.loadtxt('y_data_set')
x_new = sort_dataset(x)
y_new = sort_dataset(y)
print(np.shape(sort_dataset(x)))
print(y_new)
print(x_new)