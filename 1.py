import numpy as np


def calculate_node_flow(start_nodes, end_nodes, edge_flow):
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

st= [0, 0, 1, 2]
end = [1, 2, 0, 1]
ef = [2, 3, 2, 1]

print(calculate_node_flow(st,end, ef))